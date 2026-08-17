# agent.py
import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

DB_PATH = "./chroma_db"

SYSTEM_TEMPLATE = """You are an official University FAQ AI Assistant.
Answer the question based strictly and ONLY on the provided context. 
If the answer is not contained within the provided context, strictly state:
"Information not available in the provided university documents."

Context:
{context}

Question:
{question}

Answer:"""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def start_interactive_agent():
    if not os.path.exists(DB_PATH):
        print("Error: Vector database not found. Please run 'python ingest.py' first.")
        return

    print("Initializing University FAQ AI Agent...")
    
    # Load Embeddings and Vector DB
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    # Load Local LLM
    llm = ChatOllama(model="llama3.2", temperature=0)
    prompt = ChatPromptTemplate.from_template(SYSTEM_TEMPLATE)
    
    # Build RAG Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("\n" + "=" * 50)
    print("  University FAQ AI Assistant (Interactive Mode)")
    print("  Type 'exit' or 'quit' to stop.")
    print("=" * 50 + "\n")

    while True:
        try:
            user_question = input("\nAsk a question: ").strip()
            
            if not user_question:
                continue
                
            if user_question.lower() in ["exit", "quit"]:
                print("Exiting FAQ Assistant. Goodbye!")
                break

            print("\nSearching documents and generating answer...")
            response = rag_chain.invoke(user_question)
            
            print("\n------------------- Answer -------------------")
            print(response)
            print("----------------------------------------------")
            
        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting...")
            break

if __name__ == "__main__":
    start_interactive_agent()