#!/bin/bash
set -e

# 가상환경 생성
python3 -m venv fske-venv
source fske-venv/bin/activate

# pip 업그레이드 및 필수 휠 설치
pip install --upgrade pip wheel

# requirements 설치
pip install -r /workspace/fske/requirements.txt

# ipykernel 커널 등록 (필요 시)
python -m ipykernel install --user --name=fske-venv --display-name "Python (fske)"

# 모델 캐시 경로 안내
echo "[INFO] 모델은 반드시 /workspace/model-cache/ 디렉토리에 저장되어 있어야 함"
echo "[INFO] 예시 경로: /workspace/model-cache/beomi/gemma-ko-7b"

# 최초 실행 시 모델 다운로드 예시 (온라인 상황에서만)
# transformers-cli download beomi/gemma-ko-7b --cache-dir /workspace/model-cache
