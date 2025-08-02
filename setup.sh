#!/bin/bash
set -e

echo "[1/4] Python 가상환경 생성..."
python3 -m venv fske-venv
source fske-venv/bin/activate

echo "[2/4] pip 업그레이드 및 필수 툴 설치..."
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt

echo "[3/4] 디렉토리 준비..."
mkdir -p data
mkdir -p outputs
mkdir -p logs

echo "[4/4] 설치 완료. 다음 명령어로 가상환경을 활성화하세요:"
echo "source fske-venv/bin/activate"