import os
import base64
import zipfile
import io
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
import fitz  # PyMuPDF
import pandas as pd
from werkzeug.security import check_password_hash, generate_password_hash

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Replace with a secure key
api_key = "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"
file_uploaded = False

# Admin credentials (for simplicity, use hardcoded credentials or retrieve from DB)
ADMIN_USERNAME = 'admin'
ADMIN_PASSWORD_HASH = generate_password_hash('Buch$$2024')  # Replace with your hashed password

class Document:
    def __init__(self, text, metadata=None):
        self.page_content = text
        self.metadata = metadata if metadata is not None else {}

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
        print(f"Error processing PDF file: {e}")
        return []

# Function to load CSV and convert to text
def load_csv(file_stream):
    try:
        df = pd.read_csv(file_stream)
        return [df.to_string()]
    except Exception as e:
        print(f"Error processing CSV file: {e}")
        return []

# Function to extract and process files from ZIP
def process_zip(uploaded_file):
    docs = []
    with zipfile.ZipFile(uploaded_file) as z:
        for file_name in z.namelist():
            with z.open(file_name) as file_stream:
                texts = load_text(file_stream, file_name)
                docs.extend([Document(text) for text in texts])
    return docs

# Route for the home page
@app.route('/')
def index():
    # Redirect to admin login if admin is not logged in
    if 'admin_logged_in' not in session:
        return redirect(url_for('admin_login'))

    # If admin is logged in but no files uploaded yet, show admin dashboard
    if not file_uploaded:
        return redirect(url_for('admin_dashboard'))

    # If files have been uploaded, show the chat page for users
    return redirect(url_for('user_chat'))

# Admin login route
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == ADMIN_USERNAME and check_password_hash(ADMIN_PASSWORD_HASH, password):
            session['admin_logged_in'] = True
            return redirect(url_for('admin_dashboard'))
        else:
            return render_template('admin_login.html', error="Invalid credentials")
    return render_template('admin_login.html')

# Admin dashboard (file upload page)
@app.route('/admin')
def admin_dashboard():
    if 'admin_logged_in' not in session:
        return redirect(url_for('admin_login'))
    
    # Show the admin upload page
    return render_template('admin.html')

# Logout route for admin
@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('admin_login'))

# Route to process the uploaded file
@app.route('/upload', methods=['POST'])
def upload_file():
    global file_uploaded
    if 'admin_logged_in' not in session:
        return jsonify({'error': 'Unauthorized access!'}), 403

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded!'}), 400

    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return jsonify({'error': 'No selected file!'}), 400

    docs = process_zip(uploaded_file)
    if not docs:
        return jsonify({'error': 'No documents found in the ZIP file!'}), 400

    # Create vector store
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    model_name = "all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # Store retriever in session
    app.config['retriever'] = retriever
    file_uploaded = True
    
    # Redirect to user chat page
    return redirect(url_for('user_chat'))

# Route for querying the documents (for users to chat)
@app.route('/ask', methods=['POST'])
def ask_question():
    question = request.json.get('question')
    if 'retriever' not in app.config:
        return jsonify({'error': 'No documents uploaded!'}), 400

    retriever = app.config['retriever']
    
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
    
    return jsonify({'answer': answer}), 200

# Route for users to access the chat
@app.route('/user_chat')
def user_chat():
    if not file_uploaded:
        return "Files not uploaded yet. Please wait for the admin to upload the files."
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
