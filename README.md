# FSKE: 금융보안 QA 생성

## 프로젝트 개요
금융보안 분야의 질문에 대한 답변을 생성하기 위해 대규모 언어 모델(LLM)과 LoRA 미세튜닝을 결합한 프로젝트입니다. 기본 모델로는 **gemma-ko-7b**를 사용하며, 금융 보안 문항에 특화된 데이터를 활용하여 성능을 향상시켰습니다.

## 데이터
- `data/train.csv`, `data/test.csv`, `data/sample_submission.csv` 형태의 CSV 파일을 사용합니다.
- 모든 경로는 상대경로를 사용하여 재현성을 높였습니다.
- 외부 데이터는 사용하지 않았으며, 제공된 FSKU 데이터만 활용했습니다.

## 모델
- **gemma-ko-7b** (Google, 2024, Apache-2.0 license)를 기반으로 합니다.
- 미세튜닝은 LoRA 방법을 사용하여 메모리와 연산량을 절약하면서도 높은 성능을 얻습니다.

## 환경 세팅
- Python 3.10, CUDA 11.8 환경에서 테스트되었습니다.
- 필요한 패키지는 `requirements.txt`에 명시되어 있으며, 아래 명령어로 설치할 수 있습니다:
  ```bash
  pip install -r requirements.txt
  ```
- 가상환경 설정 및 의존성 설치를 자동화하려면 `setup.sh` 스크립트를 사용할 수 있습니다:
  ```bash
  chmod +x setup.sh
  ./setup.sh
  ```

## 훈련 방법
- 데이터 전처리: `src/preprocess.py`
- LoRA 미세튜닝: `src/finetune_lora.py --config src/config.json`
- 주요 하이퍼파라미터는 `src/config.json`에서 관리하며, 기본값은 학습률 `2e-4`, 에폭 `3`입니다.

## 프롬프트 설계
- **객관식 예시**:
  ```
  질문: 금융사고 예방을 위한 가장 효과적인 방법은?
  1. 보안 교육
  2. 시스템 업그레이드
  3. 고객 상담
  4. 광고 강화
  Answer:
  ```
- **주관식 예시**:
  ```
  질문: 개인정보보호법의 주요 목적을 서술하시오.
  Answer:
  ```

## 추론 사용법
다음 명령어로 추론을 실행하여 `submission.csv`를 생성합니다:
```bash
python src/inference.py --model_dir models/finetuned_model --test data/test.csv --output submission.csv
```

## 파일 구조
```
.
├── README.md
├── requirements.txt
├── setup.sh
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── models/
│   ├── base_model/
│   └── finetuned_model/
└── src/
    ├── config.json
    ├── finetune_lora.py
    ├── inference.py
    ├── preprocess.py
    └── utils.py
```

## 결과 및 평가
- 리더보드 점수: 추후 업데이트 예정
- 주관식 답변의 정확성과 일관성이 향후 개선 포인트입니다.

## 대회 규칙 준수 사항
- 모든 모델과 데이터는 라이선스를 준수합니다.
- 외부 API 없이 로컬 환경에서만 추론합니다.
- 단일 LLM(gemma-ko-7b)만을 사용했습니다.
