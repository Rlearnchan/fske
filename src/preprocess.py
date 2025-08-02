"""Data preprocessing utilities for Financial Security QA dataset."""

from pathlib import Path
import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """Load CSV data from path."""
    return pd.read_csv(path)

def preprocess(input_csv: str, output_csv: str) -> None:
    """Transform raw CSV into training-ready format."""
    df = load_data(input_csv)
    # TODO: Add preprocessing steps.
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess dataset")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    preprocess(args.input, args.output)
