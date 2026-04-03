from app.retrieval.dense_retriever import DenseRetriever
from app.generation.prompt_builder import build_messages
from app.generation.llm_client import chat_completion
from app.config.settings import settings

class RagAnswerer:
    def __init__(self):
        self.retriever = DenseRetriever()

    def answer(self, question: str):
        chunks = self.retriever.retrieve(question, top_k=settings.final_context_k)
        messages = build_messages(question, chunks)
        answer = chat_completion(messages)
        return {
            "question": question,
            "answer": answer,
            "chunks": chunks,
        }