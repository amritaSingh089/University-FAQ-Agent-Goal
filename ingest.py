# ingest.py
import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

DATA_PATH = "./data"
DB_PATH = "./chroma_db"

def build_vector_db():
    if not os.path.exists(DATA_PATH) or not os.listdir(DATA_PATH):
        print(f"Error: No files found in '{DATA_PATH}'. Please add your PDF or TXT files first.")
        return

    print("Loading documents from 'data' directory...")
    documents = []
    
    # Safely load PDF files individually
    for root, _, files in os.walk(DATA_PATH):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith('.pdf'):
                try:
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())
                    print(f"Successfully loaded PDF: {file}")
                except Exception as e:
                    print(f"Skipping invalid/corrupted PDF '{file}': {e}")
                    print(f"--> Tip: If '{file}' is a plain text file, rename its extension to .txt")
    
    # Load TXT files
    try:
        txt_loader = DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader)
        txt_docs = txt_loader.load()
        documents.extend(txt_docs)
        if txt_docs:
            print(f"Successfully loaded {len(txt_docs)} TXT files.")
    except Exception as e:
        print(f"Error loading TXT files: {e}")

    if not documents:
        print("No valid PDF or TXT documents were successfully loaded.")
        return

    print(f"\nLoaded {len(documents)} document pages/files in total. Chunking text...")
    
    # Split text into overlapping chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = text_splitter.split_documents(documents)
    
    print(f"Created {len(chunks)} chunks. Generating embeddings...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # Create or update vector store
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    print("SUCCESS: Documents indexed and saved to ChromaDB!")

if __name__ == "__main__":
    build_vector_db()