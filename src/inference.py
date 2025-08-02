"""Inference script for Financial Security QA model."""

from pathlib import Path
import pandas as pd

def run_inference(model_dir: str, test_path: str, output_path: str) -> None:
    """Run inference and save submission file."""
    # TODO: load model and tokenizer, generate predictions
    test_df = pd.read_csv(test_path)
    submission = pd.DataFrame({"id": test_df.get("id", range(len(test_df))),
                               "answer": [""] * len(test_df)})
    submission.to_csv(output_path, index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run inference")
    parser.add_argument("--model_dir", default="models/finetuned_model")
    parser.add_argument("--test", default="data/test.csv")
    parser.add_argument("--output", default="submission.csv")
    args = parser.parse_args()
    run_inference(args.model_dir, args.test, args.output)
