from sentence_transformers import SentenceTransformer
from app.config.settings import settings # Singleton

class Embedder:
    def __init__(self, model_name: str | None = None):
        self.model = SentenceTransformer(model_name or settings.embed_model)
        
        # BGE models require this prefix for asymmetric retrieval
        self.query_instruction = "Represent this sentence for searching relevant passages: "
        
    def encode_texts(self, texts: list[str], batch_size: int = 16):
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True
        )
    
    def encode_query(self, query: str):
        # prepend instruction to user's query before encoding
        formatted_query = f"{self.query_instruction}{query}"
        
        return self.model.encode([formatted_query], normalize_embeddings=True)[0]