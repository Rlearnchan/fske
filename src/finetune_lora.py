"""LoRA fine-tuning script for gemma-ko-7b."""

from pathlib import Path

def train(config_path: str) -> None:
    """Train model with LoRA adapters."""
    # TODO: implement training using transformers and peft
    raise NotImplementedError

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune model with LoRA")
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    train(args.config)
