#!/bin/bash
set -e

# 가상환경 생성
python3 -m venv fske-venv
source fske-venv/bin/activate

# pip 업그레이드 및 필수 휠 설치
pip install --upgrade pip wheel

# requirements 설치
pip install -r /fske/requirements.txt

# ipykernel 커널 등록 (필요 시)
python -m ipykernel install --user --name=fske-venv --display-name "Python (fske)"