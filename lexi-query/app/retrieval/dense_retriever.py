import chromadb
from app.config.settings import settings
from app.retrieval.embedder import Embedder
from app.retrieval.vector_store import COLLECTION_NAME

class DenseRetriever:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.chroma_dir)
        self.collection = self.client.get_collection(COLLECTION_NAME)
        self.embedder = Embedder()
        
    def retrieve_single_query(self, query: str, top_k: int):
        
        # 1. convert user prompt to vector space + BGE prefix
        # ChromaDB expects python list instead of numpy array
        q = self.embedder.encode_query(query).tolist()
        
        # 2. use chromadb search
        results = self.collection.query(
            query_embeddings=[q], # list to handle multiple queries, in this eg only 1
            n_results=top_k or settings.top_k
        )
        
        chunks = []
        # 3. Format results from columnar -> standard
        # idx 0 because only single prompt
        for i in range(len(results["ids"][0])):
            chunks.append({
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            })
        
        return chunks