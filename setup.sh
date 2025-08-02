#!/bin/bash
set -e

echo "🔧 Updating packages..."
sudo apt update && sudo apt install -y git wget unzip python3-pip

echo "🐍 Creating virtual environment..."
python3 -m venv fske-venv
source fske-venv/bin/activate

echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Setup complete. Run with: source fske-venv/bin/activate"
