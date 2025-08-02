#!/bin/bash
set -e
python3 -m venv fske-venv
source fske-venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt