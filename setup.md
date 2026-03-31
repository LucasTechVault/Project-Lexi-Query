1. Setup directory

```
mkdir lexi-query
cd lexi-query

mkdir -p app/{common,config,ingest,retrieval,generation,evaluation,api}
mkdir -p data/{raw,parsed,processed}
mkdir -p storage/{chroma,cache}
mkdir -p scripts
touch README.md .env.example
```

2. Setup Python Env

```
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install \
  datasets \
  sentence-transformers \
  chromadb \
  pymupdf \
  rank-bm25 \
  fastapi \
  uvicorn \
  openai \
  httpx \
  pydantic \
  pydantic-settings \
  python-dotenv \
  pandas \
  pyarrow \
  tqdm \
  tenacity \
  tiktoken \
  scikit-learn \
  rich \
  typer

```

3. Create config layer

```
APP_ENV=dev

# Local storage
CHROMA_DIR=storage/chroma
DATA_DIR=data

# Embedding model
EMBED_MODEL=BAAI/bge-large-en-v1.5

# Remote vLLM endpoint
VLLM_BASE_URL=http://<REMOTE_H100_HOST>:8000/v1
VLLM_API_KEY=token-abc123
VLLM_MODEL=meta-llama/Llama-3.3-70B-Instruct

# Retrieval
TOP_K=12
FINAL_CONTEXT_K=5
CHUNK_SIZE=800
CHUNK_OVERLAP=120
```

4. Create /app/config/settings.py
