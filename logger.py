import datetime

# This is the file where all the interactions will be saved
LOG_FILE = "rag_logs.txt"

def log_interaction(question, docs, answer, timestamps):
    """
    Logs the user interaction, retrieved chunks, and generated answer 
    to a backend text file with exact timestamps for each stage.
    """
    # Open the text file in "append" mode so we add to it without overwriting previous logs
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write(f"[{timestamps['asked']}] USER QUESTION: {question}\n\n")
        
        f.write(f"[{timestamps['retrieved']}] RETRIEVED CONTEXT CHUNKS:\n")
        for i, doc in enumerate(docs):
            f.write(f"   [Chunk {i+1}]: {doc.page_content}\n")
            
        f.write(f"\n[{timestamps['generated']}] GENERATION PROCESS COMPLETED.\n")
        f.write(f"[{timestamps['output']}] FINAL OUTPUT SHOWN: {answer}\n")
        f.write("="*80 + "\n\n")
