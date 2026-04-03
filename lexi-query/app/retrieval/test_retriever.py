from app.retrieval.dense_retriever import DenseRetriever

def run_test():
    print("[System] Loading Embedder and connecting to ChromaDB...")
    retriever = DenseRetriever()
    
    # You can change this question to test different financial topics
    query = "What were the primary risk factors regarding supply chain disruptions?"
    print(f"\n🔍 Searching for: '{query}'\n" + "="*70)
    
    # We will just pull the top 3 chunks so it doesn't flood your terminal
    chunks = retriever.retrieve_single_query(query, top_k=3)
    
    for i, chunk in enumerate(chunks, 1):
        print(f"---RESULT {i} ---")
        print(f"Source Document : {chunk['metadata']['doc_id']}")
        print(f"Page Number     : {chunk['metadata']['page_num']}")
        print(f"Distance Score  : {chunk['distance']:.4f} (Lower is better)")
        print(f"Text Snippet    :\n{chunk['text'][:300]}...\n") # Printing first 300 chars
        print("-" * 70 + "\n")

if __name__ == "__main__":
    run_test()