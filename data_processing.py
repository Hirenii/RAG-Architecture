from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_pdf_text(pdf_file):
    """
    Reads the uploaded PDF file page by page and extracts the raw text.
    
    Args:
        pdf_file: The uploaded PDF file object from Streamlit.
        
    Returns:
        str: The complete extracted text from the PDF.
    """
    text = ""
    # We use pypdf's PdfReader to read the file
    pdf_reader = PdfReader(pdf_file)
    
    # Loop through each page and extract text
    for page_num, page in enumerate(pdf_reader.pages):
        page_text = page.extract_text()
        if page_text:
            text += page_text
            
    return text

def get_text_chunks(text):
    """
    Splits the large text into smaller chunks. 
    This is important because LLMs have a limit on how much text they can process at once.
    
    Args:
        text: The complete text string.
        
    Returns:
        list: A list of text chunks (strings).
    """
    # RecursiveCharacterTextSplitter splits text by paragraphs, then sentences, etc.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,   # Maximum size of each chunk
        chunk_overlap=200, # Overlap between chunks so we don't lose context between cuts
        length_function=len
    )
    
    chunks = text_splitter.split_text(text)
    return chunks
