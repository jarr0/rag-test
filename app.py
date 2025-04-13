import traceback

import streamlit as st

# Try importing FAISS. If it fails, prompt to install faiss-cpu.
try:
    from langchain_community.vectorstores import FAISS
except ImportError as e:
    st.error("FAISS import error. Please install it with 'pip install faiss-cpu' (or use conda with Miniforge for native M1 support).")
    st.text(traceback.format_exc())
    st.stop()  # Stop execution if FAISS is not available.

from langchain.chains import RetrievalQA
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_experimental.text_splitter import SemanticChunker

# Streamlit UI
st.title("📄 RAG System with DeepSeek R1 & Ollama")

uploaded_file = st.file_uploader("Upload your PDF file here", type="pdf")

if uploaded_file:
    # Save uploaded file locally
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())

    # Load PDF document
    loader = PDFPlumberLoader("temp.pdf")
    docs = loader.load()

    # Split documents into semantic chunks
    text_splitter = SemanticChunker(HuggingFaceEmbeddings())
    documents = text_splitter.split_documents(docs)

    # Generate embeddings and index using FAISSy
    
    embedder = HuggingFaceEmbeddings()
    vector = FAISS.from_documents(documents, embedder)
    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Use Ollama as the LLM for generating answers
    llm = Ollama(model="mistral")

    # Prompt template for QA chain
    prompt = """
    Use the following context to answer the question.
    Context: {context}
    Question: {question}
    Answer:"""

    QA_PROMPT = PromptTemplate.from_template(prompt)

    # Setup chain components
    llm_chain = LLMChain(llm=llm, prompt=QA_PROMPT)
    combine_documents_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")

    # Assemble retrieval QA chain
    qa = RetrievalQA(combine_documents_chain=combine_documents_chain, retriever=retriever)

    # User input for question
    user_input = st.text_input("Ask a question about your document:")

    if user_input:
        # Get the answer from the chain and display it
        response = qa(user_input)["result"]
        st.write("**Response:**")
        st.write(response)
