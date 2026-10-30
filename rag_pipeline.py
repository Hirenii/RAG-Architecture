from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from langchain_classic.chains.question_answering import load_qa_chain

# Configuration for our models
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/flan-t5-small"

def create_vector_store(text_chunks):
    """
    Converts the text chunks into mathematical vectors (embeddings) 
    and stores them in a local FAISS database.
    
    Args:
        text_chunks (list): List of text strings.
        
    Returns:
        FAISS: The initialized vector store containing our documents.
    """
    # 1. Initialize the embedding model. This converts text into numbers.
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    # 2. Store these embeddings in FAISS (Facebook AI Similarity Search)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_conversational_chain():
    """
    Sets up the language model and the LangChain QA chain.
    
    Returns:
        chain: The configured LangChain QA chain ready to answer questions.
    """
    # 1. Load the open-source language model pipeline
    pipe = pipeline("text2text-generation", model=LLM_MODEL_NAME, max_length=512)
    
    # 2. Wrap it so LangChain can use it
    llm = HuggingFacePipeline(pipeline=pipe)
    
    # 3. Load the QA chain (stuffing strategy puts all relevant docs into the prompt)
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain

def get_answer(vector_store, user_question):
    """
    Combines retrieval and generation to answer a question.
    
    Args:
        vector_store (FAISS): Our database of document embeddings.
        user_question (str): The question asked by the user.
        
    Returns:
        str: The generated answer.
    """
    # 1. Retrieve the most relevant chunks from the database
    # We ask for the top 3 (k=3) most similar chunks to the question
    docs = vector_store.similarity_search(user_question, k=3)
    
    # 2. Get the LLM chain
    chain = get_conversational_chain()
    
    # 3. Pass the chunks and the question to the model to generate the answer
    response = chain.invoke(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    
    return response["output_text"]
