import streamlit as st
import os
import time
import pickle
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ API key is missing! Please check your .env file.")
    st.stop()

os.environ["GROQ_API_KEY"] = groq_api_key

llm = ChatGroq(api_key=groq_api_key, model_name="llama-3.3-70b-versatile")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    Make sure to give complete and easy to understand output
    <context>
    {context}
    </context>
    Question: {input}
    """
)

# ---------- Custom cleaning/processing beyond loader ----------
def custom_clean(text: str) -> str:
    """
    Simple custom cleaning:
    - Remove extra spaces
    - Lowercase text
    - Remove unwanted special chars except basic punctuation
    """
    import re
    text = text.lower()
    text = re.sub(r'[^a-z0-9.,;:?!\s]', ' ', text)
    text = ' '.join(text.split())
    return text
# ---------------------------------------------------------------


# ---------- Caching embeddings and vectors ------------------
@st.cache_resource(show_spinner=False)
def create_vector_embedding(directory_path):
    """
    Cached vector creation to avoid re-running for same files.
    """
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

    if not os.path.exists(directory_path):
        st.error(f"Directory `{directory_path}` does not exist.")
        return None

    loader = PyPDFDirectoryLoader(directory_path)
    docs = loader.load()

    if not docs:
        st.error("No PDF files found in the directory.")
        return None

    # Apply custom cleaning to each document text  (Point 1 applied here)
    for doc in docs:
        doc.page_content = custom_clean(doc.page_content)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    final_documents = text_splitter.split_documents(docs[:50])

    if not final_documents:
        st.error("Failed to split documents.")
        return None

    vectors = FAISS.from_documents(final_documents, embeddings)
    st.session_state.vectors = vectors
    return vectors
# ---------------------------------------------------------------


uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

directory_path = r"C:\Users\A.JASWANTH\OneDrive\Desktop\Rubixe_Rag_Task\research_papers"

if uploaded_files:
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    for uploaded_file in uploaded_files:
        file_path = os.path.join(directory_path, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    st.success(f"Uploaded {len(uploaded_files)} file(s) successfully!")

if st.button("Document Embedding"):
    vectors = create_vector_embedding(directory_path)
    if vectors:
        st.session_state.vectors = vectors
        st.success("Vector database initialized successfully.")

        # ---------------- Save FAISS index for API ----------------
        faiss_path = os.path.join(directory_path, "faiss_index") 
        os.makedirs(faiss_path, exist_ok=True)
        vectors.save_local(faiss_path) 


user_prompt = st.text_input("Enter your query from the research paper")

if user_prompt:
    if "vectors" in st.session_state:
        document_chain = create_stuff_documents_chain(llm, prompt)

        # ---------- Source ranking & re-ranking ----------
        # Using a compression retriever to re-rank and select the most relevant chunks
        base_retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 10})
        compressor = LLMChainExtractor.from_llm(llm)
        retriever = ContextualCompressionRetriever(base_compressor=compressor,
                                                   base_retriever=base_retriever)
        # -----------------------------------------------------

        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start_time = time.time()
        response = retrieval_chain.invoke({'input': user_prompt})
        end_time = time.time()

        st.write(f"Response time: {end_time - start_time:.2f} seconds")
        st.write(response['answer'])

        with st.expander("Document similarity search"):
            for i, doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write('-------------------')
    else:
        st.warning("Vector database is not initialized. Please click 'Document Embedding' first.")
