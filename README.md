# Rubixe_RAG_Project

# HR Policy Q&A Chatbot (RAG-based Retrieval System)

The RAG system retrieves relevant chunks of documents using embeddings and generates natural language answers using a large language model.

**Key Features:**
- Upload PDFs via Streamlit UI.
- Create embeddings and vector store for fast retrieval.
- Use Groq LLM to generate answers.
- Source ranking & re-ranking for relevance.
- Caching embeddings to avoid recomputation.
- Flask API endpoint for programmatic queries.


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
python -m streamlit run app.py
```

## How It Works

- Upload PDFs in Streamlit → Documents are cleaned → Split into chunks.
- Embed chunks using HuggingFaceEmbeddings → Store in FAISS vector database.
- User query → Vector store searches top k relevant chunks.
- Re-ranking & compression → Only the most relevant chunks are passed to the LLM.
- LLM generates answer → Displayed in Streamlit or returned via Flask API.

## Usage

### Document Embedding Process

- Enter the path to your PDF directory in the input box.
- Click "Document Embedding" to process documents.
- It will generate vector embeddings using FAISS.

### Ask Questions

- Enter a query related to the HR Policy file.
- The model retrieves relevant text & generates a detailed response.
- Check the document similarity search section for context.
