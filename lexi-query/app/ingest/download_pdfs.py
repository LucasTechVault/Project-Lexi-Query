from pathlib import Path
import pandas as pd
import httpx
from tqdm import tqdm

RAW_PARQUET = "data/raw/financebench.parquet" # read from parquet to get links
PDF_DIR = Path("data/raw/pdfs") # to save pdfs

def sanitize_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)

def download_file(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    with httpx.stream("GET", url, timeout=60.0, follow_redirects=True) as r:
        r.raise_for_status()
        
        with open(dest, 'wb') as f:
            for chunk in r.iter_bytes():
                f.write(chunk)

def main():
    df = pd.read_parquet(RAW_PARQUET)
    
    records = (
        df[["doc_name", "doc_link"]]
        .dropna()
        .drop_duplicates()
        .to_dict(orient="records")
    )
    
    for rec in tqdm(records):
        fname = sanitize_filename(rec["doc_name"] + ".pdf")
        dest = PDF_DIR / fname
        
        if dest.exists():
            continue
        
        try:
            download_file(rec["doc_link"], dest)
        except Exception as e:
            print(f"Failed {rec["doc_link"]}: {e}")

if __name__ == "__main__":
    main()