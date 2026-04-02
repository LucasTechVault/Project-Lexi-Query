import chromadb
import pandas as pd
from app.config.settings import settings
from app.retrieval.embedder import Embedder

COLLECTION_NAME = "financebench_dense"

def build_index(chunks_path: str = "data/processed/chunks.parquet"):
    client = chromadb.PersistentClient(path=settings.chroma_dir)
    collection = client.get_or_create_collection(COLLECTION_NAME)
    
    df = pd.read_parquet(chunks_path)
    embedder = Embedder()
    
    batch_size = 64
    for start in range(0, len(df), batch_size):
        cur_batch = df.iloc[start : start + batch_size]
        texts = cur_batch["text"].tolist()
        embeddings = embedder.encode_texts(texts).tolist()
        
        collection.add(
            ids=cur_batch["chunk_id"].tolist(),
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {
                    "doc_id": r.doc_id,
                    "page_num": int(r.page_num),
                    "chunk_index": int(r.chunk_index),
                    "source_file": r.source_file
                } for r in cur_batch.itertuples()
            ]
        )

if __name__ == "__main__":
    build_index()