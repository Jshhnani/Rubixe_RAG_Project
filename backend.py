from flask import Flask, request, jsonify
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import pickle

# ---------- Flask API endpoint ----------
app = Flask(__name__)

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
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

# Load the FAISS index previously created by Streamlit app
# (Assume the index is saved to disk after embedding in rag_app.py)
faiss_path = r"C:\Users\A.JASWANTH\OneDrive\Desktop\Rubixe_Rag_Task\research_papers"
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

def load_vectors():
    with open(faiss_path, "rb") as f:
        return pickle.load(f)

@app.route("/ask", methods=["POST"])
def ask_question():
    """
    POST JSON:
    {
      "query": "your question here"
    }
    """
    data = request.get_json()
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "Query is missing"}), 400

    vectors = FAISS.load_local(faiss_path, embeddings=embeddings,allow_dangerous_deserialization=True )

    document_chain = create_stuff_documents_chain(llm, prompt)

    # Use same ranking & re-ranking strategy (Point 2) as in app
    base_retriever = vectors.as_retriever(search_kwargs={"k": 10})
    compressor = LLMChainExtractor.from_llm(llm)
    retriever = ContextualCompressionRetriever(base_compressor=compressor,
                                               base_retriever=base_retriever)

    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({'input': query})

    return jsonify({
        "answer": response['answer'],
        "context": [doc.page_content for doc in response['context']]
    })

if __name__ == "__main__":
    app.run(debug=True)
# --------------------------------------------------------------
