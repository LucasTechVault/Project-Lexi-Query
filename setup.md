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

3. Create config layer (.env)
4. Create /app/config/settings.py
