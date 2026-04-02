from pathlib import Path
import fitz  # PyMuPDF
import pandas as pd

PDF_DIR = Path("data/raw/pdfs")
OUT_PATH = Path("data/parsed/pages.parquet")

def parse_pdf(pdf_path: Path):
    doc = fitz.open(pdf_path)
    rows = []
    for page_idx, page in enumerate(doc):
        text = page.get_text("text")
        rows.append({
            "doc_id": pdf_path.stem,
            "source_file": str(pdf_path),
            "page_num": page_idx + 1,
            "text": text
        })
    return rows

def main():
    all_rows = []
    for pdf_path in PDF_DIR.glob("*.pdf"):
        all_rows.extend(parse_pdf(pdf_path))
    df = pd.DataFrame(all_rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"Saved {len(df)} pages to {OUT_PATH}")

if __name__ == "__main__":
    main()