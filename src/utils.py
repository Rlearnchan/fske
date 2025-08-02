"""Utility functions for Financial Security QA project."""

from typing import Iterable

def build_prompt(question: str, options: Iterable[str] | None = None) -> str:
    """Create prompt for LLM given question and optional choices."""
    if options:
        opts = "\n".join(f"{idx}. {opt}" for idx, opt in enumerate(options, 1))
        return f"{question}\n{opts}\nAnswer:"
    return f"{question}\nAnswer:"
