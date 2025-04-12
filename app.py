import gradio as gr
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_prompty import create_chat_prompt
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import LLMChain
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from transformers import AutoTokenizer, pipeline, AutoModelForQuestionAnswering, BitsAndBytesConfig,AutoModelForCausalLM, AutoConfig
from langchain import HuggingFacePipeline
from langchain.schema import Document
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from transformers import StoppingCriteria, StoppingCriteriaList
import os
import torch

modelPath="sentence-transformers/all-mpnet-base-v2"
model_kwargs= {'device':'cuda:0'}
encode_kwargs={'normalize_embedding':False}

embeddings= HuggingFaceEmbeddings(
    model_name=modelPath,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

from huggingface_hub import login
token_access = ""  # Add your hugging face api token

try:
    login(token=token_access)
except Exception as e:
    print(f"Login failed: {e}")
model_name= 'mistralai/Mistral-7B-Instruct-v0.1'

model_config = AutoConfig.from_pretrained(model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side= "right"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype="float16"
)
use_4bit=True
compute_dtype = getattr(torch, "float16")
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config
)

def pdf_text(pdf_docs):
    text = ""
    for doc in pdf_docs:
        doc_ = PdfReader(doc)
        for page in doc_.pages:
            page_text = page.extract_text()
            if page_text: 
                text += page_text + "\n"
    return text

# 2. Chunking
def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50
    )
    chunks=text_splitter.split_text(text)
    documents = [Document(page_content=chunk) for chunk in chunks]
    return documents

def get_vectorstore(documents):
    
    db = FAISS.from_documents(documents, embedding= embeddings)
    db.save_local("faiss_index")   

class StopOnCompleteAnswer(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        # Stop if the last token is a period/question mark/exclamation
        last_token = input_ids[0][-1].item()
        return last_token in {
            tokenizer.convert_tokens_to_ids("."),
            tokenizer.convert_tokens_to_ids("?"),
            tokenizer.convert_tokens_to_ids("!")
        }

stopping_criteria = StoppingCriteriaList([StopOnCompleteAnswer()])


text_generation = pipeline(
    model = model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.2,
    return_full_text=False,
    max_new_tokens=1500,
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id,
    stopping_criteria=stopping_criteria,
)

mistral_llm = HuggingFacePipeline(pipeline=text_generation)


def get_qa_prompt():
    prompt_template = """
    ### [INST]
    Answer the question below in a detailed but concise way. 
    - End your response naturally with proper punctuation.
    - If the answer is not in the context, say: "Not found in documents."

    Context: {context}

    Question: {question} 

    [/INST]
    Answer:"""  # Explicitly primes the model to "end" after answering
    return PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def format_docs(docs):
    if isinstance(docs, list) and hasattr(docs[0], "page_content"):
        return "\n\n".join(doc.page_content for doc in docs)
    return docs  # failsafe, in case it's already plain text


def llm_chain_(prompt):
  llm_chain= LLMChain(llm=mistral_llm, prompt=prompt)
  return llm_chain

def rag_chain_(llm_chain, retriever):
   
  rag_chain = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    }
    | llm_chain
   )
  return rag_chain


def handle_pdf_upload(pdf_files):
    try:
        print("🔄 Received PDF files:", pdf_files)

        text = pdf_text(pdf_files)
        print("📝 Text extracted")

        chunks = get_chunks(text)
        print(f"🧩 Created {len(chunks)} chunks")

        get_vectorstore(chunks)
        print("✅ Vectorstore built and persisted")

        return "✅ PDFs processed and vectorstore saved successfully!"

    except Exception as e:
        error_msg = f"❌ Error while processing PDFs: {str(e)}"
        print(error_msg)
        return error_msg


def clean_response(response):
    if isinstance(response, dict):
        text = response.get("text", "")
    else:
        text = response
    
    # Ensure the response ends with proper punctuation
    if text and text[-1] not in {".", "!", "?"}:
        text += "."
    return text

def user_question(query):
  db = FAISS.load_local("/content/faiss_index", embeddings, allow_dangerous_deserialization=True)
  retriever = db.as_retriever() 
  prompt = get_qa_prompt()
  llm_chain=llm_chain_(prompt)
  rag_chain=rag_chain_(llm_chain,retriever)
  response = rag_chain.invoke(query)
  return clean_response(response)

with gr.Blocks() as demo:
    gr.Markdown("# 🔍 Ask Questions From Multiple PDFs")

    with gr.Row():
        pdf_input = gr.File(file_types=[".pdf"], file_count="multiple", label="Upload PDFs")
        upload_btn = gr.Button("📄 Process PDFs")

    status_box = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        question_input = gr.Textbox(lines=2, placeholder="Ask a question...", label="Your Question")
        ask_btn = gr.Button("🤖 Get Answer")

    answer_box = gr.Textbox(label="Answer", lines=5)

    upload_btn.click(fn=handle_pdf_upload, inputs=pdf_input, outputs=status_box)
    ask_btn.click(fn=user_question, inputs=question_input, outputs=answer_box)

demo.launch()

