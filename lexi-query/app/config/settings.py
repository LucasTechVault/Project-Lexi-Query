from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_env: str = "dev"
    chroma_dir: str = "storage/chroma"
    data_dir: str = "data"

    embed_model: str = "BAAI/bge-large-en-v1.5"

    vllm_base_url: str
    vllm_api_key: str
    vllm_model: str = "meta-llama/Llama-3.3-70B-Instruct"

    top_k: int = 12
    final_context_k: int = 5
    chunk_size: int = 800
    chunk_overlap: int = 120

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()