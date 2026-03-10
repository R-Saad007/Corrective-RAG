import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore

def load_documents(directory_path="./docs"):
    """Loads all PDF documents from the specified directory."""
    print(f"Loading PDFs from '{directory_path}'...")
    loader = DirectoryLoader(directory_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} document pages.")
    return documents

def chunk_documents(documents):
    """Splits documents into smaller, manageable pieces for the LLM context window."""
    print("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")
    return chunks

def build_vector_store(chunks, embeddings_model, url="http://localhost:6333", collection_name="company_knowledge_base"):
    """Embeds chunks and stores them in a local Qdrant database."""
    print(f"Embedding chunks and saving to Qdrant at {url}. This might take a moment...")
    
    # QdrantVectorStore handles connecting to the DB and inserting the vectors
    vector_store = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        url=url,
        prefer_grpc=False,
        collection_name=collection_name,
        force_recreate=True # Recreates the collection so you don't get duplicates on multiple runs
    )
    print(f"Successfully saved vectors to Qdrant collection: '{collection_name}'")
    return vector_store

if __name__ == "__main__":
    # Ensure the docs directory exists before we try to read it
    os.makedirs("./docs", exist_ok=True)
    
    # Initialize the local Ollama embedding model
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434"
    )
    
    # Execute the pipeline
    docs = load_documents()
    if docs:
        chunks = chunk_documents(docs)
        build_vector_store(chunks, embeddings) 
    else:
        print("No documents found. Please add PDFs to the './docs' folder and run again.")