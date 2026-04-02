import pandas as pd
from dataclasses import dataclass

@dataclass
class ChunkConfig:
    chunk_size: int = 800
    chunk_overlap: int = 120

def naive_token_chunk(text: str, chunk_size: int, overlap: int):
    words = text.split()
    start = 0
    chunks = []
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks

def build_chunks(
    pages_path: str = "data/parsed/pages.parquet",
    out_path: str = "data/processed/chunks.parquet",
    chunk_size: int = 800,
    overlap: int = 120,
):
    pages = pd.read_parquet(pages_path)
    rows = []

    for _, row in pages.iterrows():
        chunks = naive_token_chunk(row["text"], chunk_size, overlap)
        for idx, chunk in enumerate(chunks):
            rows.append({
                "chunk_id": f"{row['doc_id']}_p{row['page_num']}_c{idx}",
                "doc_id": row["doc_id"],
                "page_num": row["page_num"],
                "chunk_index": idx,
                "text": chunk,
                "source_file": row["source_file"],
            })

    pd.DataFrame(rows).to_parquet(out_path, index=False)

if __name__ == "__main__":
    build_chunks()