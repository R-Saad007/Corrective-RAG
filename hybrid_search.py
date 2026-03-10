import os
import pickle
from langchain_community.retrievers import BM25Retriever
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from ingest_qdrant import load_documents, chunk_documents

# Define the persistent storage path for the sparse index
BM25_INDEX_PATH = "./bm25_index.pkl"

def get_dense_retriever(k=3):
    """Initializes and returns the Qdrant vector store retriever."""
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
    client = QdrantClient(url="http://localhost:6333")
    
    qdrant_store = QdrantVectorStore(
        client=client, 
        collection_name="company_knowledge_base", 
        embedding=embeddings
    )
    return qdrant_store.as_retriever(search_kwargs={"k": k})

def get_sparse_retriever(k=3, rebuild=False):
    """Loads the serialized BM25 keyword index from disk, or builds it if missing."""
    if os.path.exists(BM25_INDEX_PATH) and not rebuild:
        print(f"Loading existing BM25 index from '{BM25_INDEX_PATH}'...")
        with open(BM25_INDEX_PATH, "rb") as f:
            bm25_retriever = pickle.load(f)
        
        # Ensure 'k' is updated to the requested limit even if loaded from disk
        bm25_retriever.k = k 
        return bm25_retriever
    
    print("Building new BM25 Keyword Index from local documents...")
    docs = load_documents("./docs")
    if not docs:
        raise ValueError("No documents found in './docs' to build BM25 index.")
    
    chunks = chunk_documents(docs)
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k
    
    print(f"Saving BM25 index to '{BM25_INDEX_PATH}'...")
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25_retriever, f)
        
    return bm25_retriever

def reciprocal_rank_fusion(dense_results, sparse_results, k=60):
    """Fuses lists of documents based on their combined rankings using RRF."""
    fused_scores = {}
    
    # Process dense (vector) results
    for rank, doc in enumerate(dense_results):
        doc_str = doc.page_content
        if doc_str not in fused_scores:
            fused_scores[doc_str] = {"doc": doc, "score": 0}
        fused_scores[doc_str]["score"] += 1 / (rank + k)
        
    # Process sparse (keyword) results
    for rank, doc in enumerate(sparse_results):
        doc_str = doc.page_content
        if doc_str not in fused_scores:
            fused_scores[doc_str] = {"doc": doc, "score": 0}
        fused_scores[doc_str]["score"] += 1 / (rank + k)
        
    
        
    # Sort documents by their new fused score in descending order
    reranked_results = [
        item["doc"] for item in sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)
    ]
    
    return reranked_results

def hybrid_search(query, k=3):
    """Production wrapper to execute parallel retrieval and fusion."""
    dense_retriever = get_dense_retriever(k=k)
    sparse_retriever = get_sparse_retriever(k=k)
    
    # In a fully asynchronous production backend (like FastAPI), 
    # these two invokes would be wrapped in asyncio.gather() to fire simultaneously.
    # For now, running sequentially is extremely fast.
    dense_docs = dense_retriever.invoke(query)
    sparse_docs = sparse_retriever.invoke(query)
    
    return reciprocal_rank_fusion(dense_docs, sparse_docs)

if __name__ == "__main__":
    # Test the production-ready retrieval pipeline
    test_query = "What does the module ClickOps do?"
    print(f"\n--- Testing Production Hybrid Search: '{test_query}' ---")
    
    final_documents = hybrid_search(test_query)
    
    if final_documents:
        print("\n--- Top Recommended Context Chunk ---")
        print(final_documents[0].page_content)
        print("\nSource Metadata:", final_documents[0].metadata)
    else:
        print("No documents retrieved.")