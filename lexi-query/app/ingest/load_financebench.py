from datasets import load_dataset
from pathlib import Path
import pandas as pd

def export_financebench(out_path: str = "data/raw/financebench.parquet") -> pd.DataFrame:
    ds = load_dataset("PatronusAI/financebench", split="train") # download from huggingface
    df = ds.to_pandas()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df

if __name__ == "__main__":
    df = export_financebench()
    print(df.columns.tolist())
    print(df.head(3).to_dict(orient="records"))