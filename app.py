import os
import base64
import zipfile
import io
import streamlit as st
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
import fitz  # PyMuPDF
import pandas as pd
from werkzeug.security import check_password_hash, generate_password_hash

# Admin credentials
ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD_HASH = generate_password_hash('Buch$$2024')  # Replace with your hashed password

# API Key for ChatGroq
api_key = "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"

# Initialize the vector store and document retriever
retriever = None

# Function to load text from various file types
def load_text(file_stream, file_name):
    if file_name.endswith('.pdf'):
        return load_pdf(file_stream)
    elif file_name.endswith('.csv'):
        return load_csv(file_stream)
    return []

# Function to load PDF and extract text
def load_pdf(file_stream):
    file_bytes = io.BytesIO(file_stream.read())
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        texts = []
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            texts.append(page.get_text("text"))
        return texts
    except Exception as e:
        st.error(f"Error processing PDF file: {e}")
        return []

# Function to load CSV and convert to text
def load_csv(file_stream):
    try:
        df = pd.read_csv(file_stream)
        return [df.to_string()]
    except Exception as e:
        st.error(f"Error processing CSV file: {e}")
        return []

# Function to process ZIP files
def process_zip(uploaded_file):
    docs = []
    with zipfile.ZipFile(uploaded_file) as z:
        for file_name in z.namelist():
            with z.open(file_name) as file_stream:
                texts = load_text(file_stream, file_name)
                docs.extend([Document(text) for text in texts])
    return docs

# Class to hold document content
class Document:
    def __init__(self, text, metadata=None):
        self.page_content = text
        self.metadata = metadata if metadata is not None else {}

# Streamlit App
def main():
    global retriever
    st.title("Admin Dashboard")

    # Admin login section
    if "admin_logged_in" not in st.session_state:
        with st.form("login_form"):
            st.write("Admin Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_button = st.form_submit_button("Login")
        
        if login_button:
            if username == ADMIN_USERNAME and check_password_hash(ADMIN_PASSWORD_HASH, password):
                st.session_state["admin_logged_in"] = True
                st.success("Successfully logged in!")
            else:
                st.error("Invalid credentials")
        return

    # File Upload Section
    st.write("Upload a ZIP file containing PDFs or CSVs")
    uploaded_file = st.file_uploader("Choose a file", type="zip")
    
    if uploaded_file is not None:
        docs = process_zip(uploaded_file)
        if docs:
            st.success(f"Successfully processed {len(docs)} documents")

            # Create vector store from documents
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            model_name = "all-MiniLM-L6-v2"
            embeddings = HuggingFaceEmbeddings(model_name=model_name)
            vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
            retriever = vectorstore.as_retriever()
            st.success("Document retriever is ready!")

    # Question and Answer Section
    st.title("User Chat")
    if retriever is not None:
        question = st.text_input("Ask a question about the documents")
        if st.button("Submit"):
            if question:
                system_prompt = (
                    "You are an assistant for question-answering tasks based on the provided/uploaded documents. "
                    "When a query is received, first search the content of the documents to find a relevant answer. If the answer is available, "
                    "provide a detailed and informative response, ensuring it is neither too lengthy nor too concise. "
                    "If the query is not addressed in the documents, then you must provide the answer based on your knowledge."
                    "\n\n{context}"
                )

                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", system_prompt),
                        ("human", question),
                    ]
                )

                llm = ChatGroq(model="llama3-70b-8192", groq_api_key=api_key)
                question_answer_chain = create_stuff_documents_chain(llm, prompt)
                rag_chain = create_retrieval_chain(retriever, question_answer_chain)

                results = rag_chain.invoke({"input": question})
                answer = results['answer']
                st.write(f"Answer: {answer}")
            else:
                st.error("Please enter a question")
    else:
        st.info("Please upload a document first")

if __name__ == '__main__':
    main()
