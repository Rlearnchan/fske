#!/bin/bash
set -e

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

# Activate the virtual environment
source .venv/bin/activate

pip install --upgrade pip wheel
pip install --no-cache-dir -r requirements.txt

echo "Virtual environment created at .venv. Activate with 'source .venv/bin/activate'."
