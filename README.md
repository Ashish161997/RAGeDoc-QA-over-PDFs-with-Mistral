# 🔍 Multi-PDF Question Answering App using Mistral & LangChain

This project is an interactive Gradio web application that allows you to upload multiple PDF files and ask questions about their content. It uses a powerful Retrieval-Augmented Generation (RAG) pipeline powered by:

- 🧠 Mistral 7B Instruct (4-bit quantized)
- 🗂️ FAISS for semantic vector search
- 📚 Sentence Transformers for embeddings
- 🔗 LangChain for chaining and prompt handling
- ⚡ Gradio for the user interface

---

## 🚀 Features

- Upload multiple PDFs
- Automatically parse and chunk PDF content
- Store and retrieve relevant chunks using FAISS
- Use a locally quantized Mistral 7B model to answer your question
- Clean and concise response generation

---

## 🧰 Requirements

Ensure you have the following installed:

```bash
pip install -r requirements.txt
```

Update the path if needed (e.g., relative or absolute) depending on where you store the faiss_index folder.
Once that’s done…
```python
python app.py
```
To run chatbot
```python
python app_chatbot.py
```

