import streamlit as st
import datetime
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# We removed HuggingFacePipeline and local transformers, now we use Google's Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
from logger import log_interaction

import os
from dotenv import load_dotenv

load_dotenv()

# 1. Local Embedding Model (Converts text to vectors locally for free)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# 2. Add your Gemini API Key here (Loaded from .env or environment)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

def get_pdf_text(pdf_file):
    """Extracts raw text from an uploaded PDF file."""
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def get_text_chunks(text):
    """
    Splits the large text into smaller chunks.
    This prevents the AI from being overwhelmed with too much information at once.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200, 
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Converts the text chunks into vectors and stores them in a local FAISS database."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_conversational_chain():
    """
    Sets up the Brain (LLM) that will answer our questions.
    We are now using Gemini 1.5 Flash via the API instead of a local model.
    """
    # Initialize the Gemini model using the API key
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=GEMINI_API_KEY, 
        temperature=0.3 # Low temperature makes the answers more factual and less creative
    )
    
    # Create a custom prompt to force the AI to answer in a consistent length
    custom_prompt_template = """
You are a helpful and professional AI assistant. 
Use the following pieces of retrieved context to answer the user's question. 
If you don't know the answer based on the context, just say that you don't know.
You MUST provide a complete, well-formed answer that is exactly 2 to 4 sentences long. Do not leave sentences unfinished.

Context:
{context}

Question: {question}

Helpful Answer:"""
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    
    # We pass the custom prompt to the LangChain QA chain
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain

def main():
    st.set_page_config(page_title="RAG Chatbot", page_icon="📚")
    
    st.title("RAG Chatbot 🤖")
    
    # Sidebar for Uploading, and Processing
    with st.sidebar:
        st.header("1. Settings & Data")
        
        pdf_file = st.file_uploader("Upload your PDF Document", type=["pdf"])
        
        if pdf_file is not None:
            if st.button("Process Document"):
                with st.spinner("Extracting text and building database..."):
                    raw_text = get_pdf_text(pdf_file)
                    text_chunks = get_text_chunks(raw_text)
                    st.session_state.vector_store = get_vector_store(text_chunks)
                st.success("Database is ready! You can now ask questions.")

    # Main chat interface (Only shows if the DB is ready)
    if "vector_store" in st.session_state:
        st.header("2. Ask Questions")

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input
        if user_question := st.chat_input("Enter your question about the PDF:"):
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(user_question)
            
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_question})

            time_asked = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                with st.spinner("Gemini is thinking..."):
                    # Step 1: Search the database for relevant chunks
                    docs = st.session_state.vector_store.similarity_search(user_question, k=3)
                    time_retrieved = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                    
                    # Step 2: Pass chunks to Gemini
                    chain = get_conversational_chain()
                    try:
                        response = chain.invoke(
                            {"input_documents": docs, "question": user_question},
                            return_only_outputs=True
                        )
                        answer = response["output_text"]
                    except Exception as e:
                        answer = f"An error occurred with the Gemini API. Please check your API key. Error: {e}"
                        
                    time_generated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                    
                    if "Question:" in answer:
                        answer = answer.split("Question:")[0].strip()
                        
                    time_output = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                    
                    timestamps = {
                        "asked": time_asked,
                        "retrieved": time_retrieved,
                        "generated": time_generated,
                        "output": time_output
                    }
                    
                    # Secretly log everything in the backend text file
                    log_interaction(user_question, docs, answer, timestamps)
                    
                    st.markdown(answer)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        st.info("Please upload and process a PDF document from the sidebar to begin.")

if __name__ == "__main__":
    main()
