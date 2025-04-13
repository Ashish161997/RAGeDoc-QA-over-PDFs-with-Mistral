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
from langchain_huggingface import HuggingFaceEndpoint# ------------------------------
# PDF-based RAG QA system with Mistral + Gradio
# ------------------------------

import os
import torch
import gradio as gr
from PyPDF2 import PdfReader

from transformers import (
    AutoTokenizer, pipeline,
    AutoModelForCausalLM, AutoConfig,
    BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain import HuggingFacePipeline
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough

from huggingface_hub import login


# ------------------------------
# Device setup for CUDA
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# Embedding model config
# ------------------------------
modelPath = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": str(device)}
encode_kwargs = {"normalize_embedding": False}

embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# ------------------------------
# Login to Hugging Face
# ------------------------------
token_access = ""  # Replace with your HF token
try:
    login(token=token_access)
except Exception as e:
    print(f"Login failed: {e}")

# ------------------------------
# Load Mistral model in 4bit
# ------------------------------
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
model_config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

# Load model on device
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# ------------------------------
# PDF Text Extraction
# ------------------------------
def pdf_text(pdf_docs):
    text = ""
    for doc in pdf_docs:
        reader = PdfReader(doc)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# ------------------------------
# Text Chunking
# ------------------------------
def get_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    return [Document(page_content=chunk) for chunk in chunks]

# ------------------------------
# Build and save FAISS vectorstore
# ------------------------------
def get_vectorstore(documents):
    db = FAISS.from_documents(documents, embedding=embeddings)
    db.save_local("faiss_index")

# ------------------------------
# Custom stopping criteria for LLM
# ------------------------------
class StopOnCompleteAnswer(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        last_token = input_ids[0][-1].item()
        return last_token in {
            tokenizer.convert_tokens_to_ids("."),
            tokenizer.convert_tokens_to_ids("?"),
            tokenizer.convert_tokens_to_ids("!")
        }

# ------------------------------
# HuggingFace pipeline with stopping
# ------------------------------
stopping_criteria = StoppingCriteriaList([StopOnCompleteAnswer()])
text_generation = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.2,
    return_full_text=False,
    max_new_tokens=1500,
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id,
    stopping_criteria=stopping_criteria,
)

# Wrap in LangChain interface
mistral_llm = HuggingFacePipeline(pipeline=text_generation)

# ------------------------------
# Prompt and Chains
# ------------------------------
def get_qa_prompt():
    prompt_template = """
    ### [INST]
    Answer the question below in a detailed but concise way. 
    - End your response naturally with proper punctuation.
    - If the answer is not in the context, say: "Not found in documents."

    Context: {context}

    Question: {question} 

    [/INST]
    Answer:"""
    return PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def format_docs(docs):
    if isinstance(docs, list) and hasattr(docs[0], "page_content"):
        return "\n\n".join(doc.page_content for doc in docs)
    return docs

def llm_chain_(prompt):
    return LLMChain(llm=mistral_llm, prompt=prompt)

def rag_chain_(llm_chain, retriever):
    return {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    } | llm_chain

# ------------------------------
# Upload handler
# ------------------------------
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

# ------------------------------
# Clean up model response
# ------------------------------
def clean_response(response):
    text = response.get("text", "") if isinstance(response, dict) else response
    if text and text[-1] not in {".", "!", "?"}:
        text += "."
    return text

# ------------------------------
# Handle question from user
# ------------------------------
def user_question(query):
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()
    prompt = get_qa_prompt()
    chain = llm_chain_(prompt)
    rag_chain = rag_chain_(chain, retriever)
    response = rag_chain.invoke(query)
    return clean_response(response)

# ------------------------------
# Gradio UI
# ------------------------------
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
