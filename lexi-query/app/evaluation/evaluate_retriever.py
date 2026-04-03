import pandas as pd
from app.retrieval.dense_retriever import DenseRetriever

def load_eval_data(filepath="data/raw/financebench.parquet", sample_size=50):
    
    # 1. load df
    df = pd.read_parquet(filepath)
    
    # 2. sample data (10 of 150)
    test_df = df.sample(n=sample_size, random_state=42)
    
    # 3. format sample for eval loop
    eval_data = []
    for _, row in test_df.iterrows():
        eval_data.append({
            "question": row["question"],
            "gold_doc_id": row["doc_name"]
        })
    
    return eval_data

def run_eval():
    print("Initializing Retriever Evaluation")
    retriever = DenseRetriever()
    top_k = 5
    
    eval_data = load_eval_data(sample_size=150)
    
    total_queries = len(eval_data)
    hits = 0
    mrr_sum = 0.
    
    # Eval Loop
    for idx, item in enumerate(eval_data, 1):
        qns = item["question"]
        gold_doc = item["gold_doc_id"]
        
        print(f"[{idx} / {total_queries}] Testing: '{qns}'")
        
        # Pull top 5 chunks
        chunks = retriever.retrieve_single_query(qns, top_k=top_k)
        
        # extract just doc_ids from metadata
        retrieved_docs = [chunk["metadata"]["doc_id"] for chunk in chunks]
        
        # calculate metric
        if gold_doc in retrieved_docs:
            hits += 1
            rank = retrieved_docs.index(gold_doc) + 1
            mrr_sum += (1. / rank)
            print(f"Hit! Correct document found at Rank {rank}")
        else:
            print(f"Miss! Expected {gold_doc}, but got {set(retrieved_docs)}")
    
    # Generate final report
    hit_rate = (hits / total_queries) * 100
    mrr = mrr_sum / total_queries
    
    print("\n" + "="*50)
    print(f"Retrieval Eval Report (Top-5)")
    print("="*50)
    print(f"Total Test Questions: {total_queries}")
    print(f"Doc Hit@5 Rate: {hit_rate:.1f}")
    print(f"Mean Reciprocal Rank: {mrr:.4f}")
    print("="*50)
    
if __name__ == "__main__":
    run_eval()