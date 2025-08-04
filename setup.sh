#!/bin/bash
set -e

# 1. Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

# 2. Activate the virtual environment
source .venv/bin/activate

# 3. Upgrade pip and install packages
pip install --upgrade pip wheel
pip install --no-cache-dir -r requirements.txt

# 4. Register the virtual environment as a Jupyter kernel
python -m ipykernel install --user --name fske --display-name "Python (.venv-fske)"

# 5. Optional: Confirm success
echo "✅ Virtual environment created at .venv"
echo "✅ Kernel 'fske' registered for Jupyter"
echo "✅ Activate manually with: source .venv/bin/activate"
