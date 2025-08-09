# 금융보안 LLM 프로젝트 (FSKE)

한국어 금융보안 분야 데이터셋 기반 LLM 모델 개발 및 평가 프로젝트

## 📁 프로젝트 구조

```
fske/
├── data/                   # 데이터셋
│   ├── CyberMetric/        # 사이버 보안 메트릭 데이터
│   ├── FinShibainu/        # 금융 시바이누 데이터
│   ├── law/                # 법령 데이터
│   └── SecBench/           # 보안 벤치마크 데이터
├── krx_llm_dataset/        # KRX LLM 데이터셋 처리
├── note/                   # 분석 및 실험 노트북
├── output/                 # 모델 결과 (4bit/8bit/BF16)
└── others/                 # 문서 및 참고자료
```

## 🚀 주요 기능

- **데이터셋 관리**: 다양한 금융보안 도메인 데이터 통합 관리
- **모델 처리**: 다양한 정밀도(4bit, 8bit, BF16) 지원
- **템플릿 시스템**: Jinja2 기반 프롬프트 템플릿
- **Few-shot 학습**: 다양한 옵션 수 지원 (4-7개)

## 📊 지원 모델

- gemma-ko-7b
- ax-4.0-light-7b
- exaone-4.0-32b
- HyperCLOVAX-SEED-Think-14B
- koalpaca-polyglot-12.8b
- midm-2.0-11.5b

## 📝 사용법

1. 데이터셋 다운로드: `note/download_data.ipynb`
2. 모델 다운로드: `note/download_model.ipynb`
3. QA 생성: `note/generate_qa_llm.ipynb`
4. 파인튜닝: `note/qlora_finetune.ipynb`

## 📄 제출 형식

결과는 `sample_submission.csv` 형식에 맞춰 `output/` 디렉토리에 저장됩니다.
