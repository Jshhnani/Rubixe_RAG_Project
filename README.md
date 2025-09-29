# Rubixe_RAG_Project

# HR Policy Q&A Chatbot (RAG-based Retrieval System)

This project is a Retrieval-Augmented Generation (RAG)-based Q&A chatbot that extracts relevant answers from HR Policy. It utilizes LangChain, FAISS, Hugging Face embeddings, and Groq's LLaMA 3.3-70B model to provide accurate responses based on uploaded PDFs.

## Features

- Upload & process PDF research papers
- Convert text into vector embeddings (FAISS)
- Retrieve relevant context using semantic search
- Generate accurate responses using LLaMA 3.3-70B
- Interactive UI with Streamlit


## Set up a virtual environment (Optional but recommended)

```bash
python -m venv venv
venv\Scripts\activate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Set up environment variables

Create a .env file in the project root and add your Groq API Key:

GROQ_API_KEY=your_groq_api_key

## Run the Streamlit app

```bash
streamlit run app.py
```

## How It Works

- Upload Research Papers: The app loads PDFs from a specified folder.
- Text Preprocessing: Splits documents into chunks using RecursiveCharacterTextSplitter.
- Vector Embedding Creation: Uses BAAI/bge-small-en embeddings for semantic similarity.
- Search & Retrieval: Uses FAISS vector search to find relevant document chunks.
- Answer Generation: Queries LLaMA 3.3-70B for a detailed response.

## Usage

### Document Embedding Process

- Enter the path to your PDF directory in the input box.
- Click "Document Embedding" to process documents.
- It will generate vector embeddings using FAISS.

### Ask Questions

- Enter a query related to the HR Policy file.
- The model retrieves relevant text & generates a detailed response.
- Check the document similarity search section for context.
