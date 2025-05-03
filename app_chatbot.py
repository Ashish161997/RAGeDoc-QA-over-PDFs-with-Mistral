import os
import torch
import gradio as gr
from PyPDF2 import PdfReader
from transformers import (
    AutoTokenizer, pipeline,
    AutoModelForCausalLM, AutoConfig,
    BitsAndBytesConfig
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain import HuggingFacePipeline

# ------------------------------
# Device setup
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
# Load Mistral model in 4bit
# ------------------------------
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
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

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# ------------------------------
# Improved Text Generation Pipeline
# ------------------------------
text_generation = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.1,
    return_full_text=False,
    max_new_tokens=2000,
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id,
)

# Wrap in LangChain interface
mistral_llm = HuggingFacePipeline(pipeline=text_generation)

# ------------------------------
# PDF Processing Functions
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

def get_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = splitter.split_text(text)
    return [Document(page_content=chunk) for chunk in chunks]

def get_vectorstore(documents):
    db = FAISS.from_documents(documents, embedding=embeddings)
    db.save_local("faiss_index")

# ------------------------------
# Conversational Prompt Template
# ------------------------------
def get_qa_prompt():
    prompt_template = """<s>[INST] 
    You are a helpful, knowledgeable AI assistant. Answer the user's question based on the provided context.
    
    Guidelines:
    - Respond in a natural, conversational tone
    - Be detailed but concise
    - Use paragraphs and bullet points when appropriate
    - If you don't know, say so
    - Maintain a friendly and professional demeanor
    
    Conversation History:
    {chat_history}
    
    Relevant Context:
    {context}
    
    Current Question: {question} 
    
    Provide a helpful response: [/INST]"""
    
    return PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question", "chat_history"]
    )

# ------------------------------
# Chat Handling Functions
# ------------------------------
def handle_pdf_upload(pdf_files):
    try:
        if not pdf_files:
            return "⚠️ Please upload at least one PDF file"
        
        text = pdf_text(pdf_files)
        if not text.strip():
            return "⚠️ Could not extract text from PDFs - please try different files"
            
        chunks = get_chunks(text)
        get_vectorstore(chunks)
        return f"✅ Processed {len(pdf_files)} PDF(s) with {len(chunks)} text chunks"
    except Exception as e:
        return f"❌ Error: {str(e)}"

def format_chat_history(chat_history):
    return "\n".join([f"User: {q}\nAssistant: {a}" for q, a in chat_history[-3:]])

def user_query(msg, chat_history):
    if not os.path.exists("faiss_index"):
        chat_history.append((msg, "Please upload PDF documents first so I can help you."))
        return "", chat_history
    
    try:
        # Load vector store
        db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        retriever = db.as_retriever(search_kwargs={"k": 3})
        
        # Get relevant context
        docs = retriever.get_relevant_documents(msg)
        context = "\n\n".join([d.page_content for d in docs])
        
        # Generate response
        prompt = get_qa_prompt()
        chain = LLMChain(llm=mistral_llm, prompt=prompt)
        
        response = chain.run({
            "question": msg,
            "context": context,
            "chat_history": format_chat_history(chat_history)
        })
        
        # Clean response
        response = response.strip()
        for end_token in ["</s>", "[INST]", "[/INST]"]:
            if response.endswith(end_token):
                response = response[:-len(end_token)].strip()
        
        chat_history.append((msg, response))
        return "", chat_history
        
    except Exception as e:
        error_msg = f"Sorry, I encountered an error: {str(e)}"
        chat_history.append((msg, error_msg))
        return "", chat_history

# ------------------------------
# Gradio Interface
# ------------------------------
with gr.Blocks(theme=gr.themes.Soft(), title="PDF Chat Assistant") as demo:
    with gr.Row():
        gr.Markdown("""
        # 📚 PDF Chat Assistant
        ### Have natural conversations with your documents
        """)
    
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### Document Upload")
            pdf_input = gr.File(
                file_types=[".pdf"],
                file_count="multiple",
                label="Upload PDFs",
                height=100
            )
            upload_btn = gr.Button("Process Documents", variant="primary")
            status_box = gr.Textbox(label="Status", interactive=False)
            gr.Markdown("""
            **Instructions:**
            1. Upload PDF documents
            2. Click Process Documents
            3. Start chatting in the right panel
            """)
        
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                height=600,
                bubble_full_width=False,
                avatar_images=(
                    "user.png", 
                    "bot.png"
                )
            )
            
            with gr.Row():
                message = gr.Textbox(
                    placeholder="Type your question about the documents...",
                    show_label=False,
                    container=False,
                    scale=7,
                    autofocus=True
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)
            
            with gr.Row():
                clear_chat = gr.Button("🧹 Clear Conversation")
                examples = gr.Examples(
                    examples=[
                        "Summarize the key points from the documents",
                        "What are the main findings?",
                        "Explain this in simpler terms"
                    ],
                    inputs=message,
                    label="Example Questions"
                )

    # Event handlers
    upload_btn.click(
        fn=handle_pdf_upload,
        inputs=pdf_input,
        outputs=status_box
    )
    
    submit_btn.click(
        fn=user_query,
        inputs=[message, chatbot],
        outputs=[message, chatbot]
    )
    
    message.submit(
        fn=user_query,
        inputs=[message, chatbot],
        outputs=[message, chatbot]
    )
    
    clear_chat.click(
        lambda: [],
        None,
        chatbot,
        queue=False
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=True,
        debug=True
    )
