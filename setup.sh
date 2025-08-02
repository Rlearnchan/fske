#!/bin/bash

# -------------------------------
# [FSKE] RunPod Setup Script
# -------------------------------

# 1. 가상환경 설정
python3 -m venv fske-venv
source fske-venv/bin/activate

# 2. pip 최신화 및 필수 라이브러리 설치
pip install --upgrade pip
pip install -r requirements.txt

# 3. HF 캐시 디렉토리 설정 (Pod Volume 활용)
export HF_HOME=/workspace/hf_home
export TRANSFORMERS_CACHE=/workspace/hf_cache
mkdir -p $HF_HOME $TRANSFORMERS_CACHE

# 4. Jupyter 커널 등록
python -m ipykernel install --user --name fske --display-name "Python (fske)"

# 5. 베이스라인 모델 다운로드 (필요 시 주석 해제)
# 예: 한국어 튜닝된 LLM, 추후 변경 가능
# model_id="beomi/gemma-ko-7b"
# python -c "
# from transformers import AutoTokenizer, AutoModelForCausalLM
# AutoTokenizer.from_pretrained('$model_id', cache_dir='$TRANSFORMERS_CACHE')
# AutoModelForCausalLM.from_pretrained('$model_id', cache_dir='$TRANSFORMERS_CACHE')
# "

# ✅ 추후 실험용 모델 예시 (용량 고려하여 한 번에 1~2개만)
# - snunlp/KULLM-1.3B
# - nlpai-lab/kullm-polyglot-5.8b
# - tinyllama
# - open-ko-llama
# - neuropark/KoT5-Base

echo "✅ [FSKE] Setup complete."
