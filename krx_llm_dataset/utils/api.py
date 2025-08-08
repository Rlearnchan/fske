""" request, GPT API 및 관련된 함수 모음"""

import os
from typing import Literal, Any, List, Dict
import json
import requests
import random
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

# OpenAI는 optional import (기존 코드 호환성을 위해)
try:
    from openai import OpenAI
    import tiktoken
    OPENAI_AVAILABLE = True
except ImportError:
    print("⚠️ OpenAI 모듈이 설치되지 않았습니다. 로컬 LLM만 사용 가능합니다.")
    OpenAI = None
    tiktoken = None
    OPENAI_AVAILABLE = False
from .utils import load_env
from .config import AGENT_LIST, PathConfig
from .template import PromptTemplates
from .datamodel import (
    MCQAOutputFormat,
    QAOutputFormat,
    QAAnswerModel,
    PreferenceOutputFormat,
    HallucinationOutputFormat,
    ValueOutputModel,
    ClassificationOutputModel,
    EnglishMCQAFromMCQAOutputFormat,
    KoreanMCQAFromMCQAOutputFormat,
    MCQAFromMCQAOutputFormat
)

# 로컬 LLM을 위한 전역 변수
_local_model = None
_local_tokenizer = None
_local_pipeline = None

def _initialize_local_llm(model_path=None):
    """로컬 LLM 모델을 초기화합니다."""
    global _local_model, _local_tokenizer, _local_pipeline
    
    if _local_model is None:
        if model_path is None:
            path_config = PathConfig()
            model_path = path_config.LOCAL_MODEL_PATH
        
        print(f"로컬 LLM 모델 로딩 중: {model_path}")
        
        # 4bit 양자화 설정
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # 토크나이저 로드
        _local_tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="left"
        )
        
        # 모델 로드 (4bit 양자화 적용)
        _local_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            trust_remote_code=True
        )
        
        # 파이프라인 생성
        _local_pipeline = pipeline(
            "text-generation",
            model=_local_model,
            tokenizer=_local_tokenizer,
            device_map="auto"
        )
        
        print("로컬 LLM 모델 로딩 완료!")
    
    return _local_pipeline

def _local_completion(user_prompt, system_prompt: str = "", response_format=None, model: str = "exaone-4.0-32b"):
    """로컬 LLM으로 Completion을 생성
    
    Args:
        user_prompt: User Prompt
        system_prompt: System Prompt
        response_format: 현재는 사용하지 않음 (로컬 LLM에서는 JSON 파싱을 별도로 처리)
        model: 로컬 모델명 (현재는 사용하지 않음)
    
    Returns: 로컬 LLM으로부터 제공받은 답변 텍스트
    """
    # 로컬 LLM 초기화
    pipe = _initialize_local_llm()
    
    # 프롬프트 구성
    if system_prompt:
        full_prompt = f"[INST] {system_prompt}\n\n{user_prompt} [/INST]"
    else:
        full_prompt = f"[INST] {user_prompt} [/INST]"
    
    # 생성
    outputs = pipe(
        full_prompt,
        max_new_tokens=2048,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=pipe.tokenizer.eos_token_id
    )
    
    # 결과 추출
    generated_text = outputs[0]["generated_text"]
    
    # 프롬프트 부분 제거하고 답변만 추출
    if "[/INST]" in generated_text:
        response_text = generated_text.split("[/INST]")[-1].strip()
    else:
        response_text = generated_text.strip()
    
    # OpenAI API 응답 형식과 유사하게 구조화
    class MockResponse:
        def __init__(self, text, response_format=None):
            self.choices = [MockChoice(text, response_format)]
    
    class MockChoice:
        def __init__(self, text, response_format=None):
            self.message = MockMessage(text, response_format)
    
    class MockMessage:
        def __init__(self, text, response_format=None):
            self.content = text
            # JSON 형식 응답 처리를 위한 parsed 속성
            if response_format:
                try:
                    # JSON 형식의 응답을 파싱 시도
                    import re
                    json_match = re.search(r'\{.*\}', text, re.DOTALL)
                    if json_match:
                        json_text = json_match.group()
                        parsed_data = json.loads(json_text)
                        
                        # Pydantic 모델로 파싱 시도
                        if hasattr(response_format, '__call__'):
                            self.parsed = response_format(**parsed_data)
                        else:
                            self.parsed = type('obj', (object,), parsed_data)
                    else:
                        # JSON을 찾지 못한 경우 기본값 생성
                        if response_format == MCQAOutputFormat:
                            self.parsed = type('obj', (object,), {
                                'questions': [type('obj', (object,), {'question': text, 'options': ['A', 'B', 'C', 'D']})],
                                'answers': [type('obj', (object,), {'reasoning_process': '', 'answer': 'A'})]
                            })
                        elif response_format == QAOutputFormat:
                            # 텍스트에서 QA 형식을 파싱 시도
                            qa_pairs = self._parse_qa_from_text(text)
                            self.parsed = type('obj', (object,), {
                                'questions': [type('obj', (object,), {'question': qa['question']}) for qa in qa_pairs],
                                'answers': [type('obj', (object,), {'answer': qa['answer']}) for qa in qa_pairs]
                            })
                        else:
                            self.parsed = type('obj', (object,), {'content': text})
                except Exception as e:
                    print(f"JSON 파싱 오류: {e}")
                    # 파싱 실패시 기본값 생성
                    if response_format == MCQAOutputFormat:
                        self.parsed = type('obj', (object,), {
                            'questions': [type('obj', (object,), {'question': text, 'options': ['A', 'B', 'C', 'D']})],
                            'answers': [type('obj', (object,), {'reasoning_process': '', 'answer': 'A'})]
                        })
                    elif response_format == QAOutputFormat:
                        # 텍스트에서 QA 형식을 파싱 시도
                        qa_pairs = self._parse_qa_from_text(text)
                        self.parsed = type('obj', (object,), {
                            'questions': [type('obj', (object,), {'question': qa['question']}) for qa in qa_pairs],
                            'answers': [type('obj', (object,), {'answer': qa['answer']}) for qa in qa_pairs]
                        })
                    else:
                        self.parsed = type('obj', (object,), {'content': text})
            else:
                self.parsed = type('obj', (object,), {'content': text})
        
        def _parse_qa_from_text(self, text):
            """텍스트에서 QA 형식을 파싱하는 메서드"""
            import re
            qa_pairs = []
            
            # 개선된 파싱 로직: 프롬프트 형식과 일치
            # 1순위: ### question: ... ### answer: ... 패턴 (프롬프트 형식)
            primary_pattern = r'### question:\s*(.*?)\s*### answer:\s*(.*?)(?=### question:|$)'
            matches = re.findall(primary_pattern, text, re.DOTALL | re.IGNORECASE)
            
            if matches:
                for question, answer in matches:
                    # 번호 제거 및 정리
                    clean_question = self._clean_text(question)
                    clean_answer = self._clean_text(answer)
                    
                    if clean_question and clean_answer:
                        qa_pairs.append({
                            'question': clean_question,
                            'answer': clean_answer
                        })
            else:
                # 백업 패턴들
                backup_patterns = [
                    # 한글 패턴
                    r'### 질문\s*:?\s*(.*?)\s*### 답변\s*:?\s*(.*?)(?=### 질문|$)',
                    # 번호가 있는 패턴들
                    r'### 질문 \d+\s*(.*?)\s*### 답변 \d+\s*(.*?)(?=### 질문 \d+|$)',
                    r'### 데이터 \d+\s*### 질문\s*(.*?)\s*### 답변\s*(.*?)(?=### 데이터 \d+|$)',
                    # 단순 분리 패턴
                    r'질문\s*:?\s*(.*?)\s*답변\s*:?\s*(.*?)(?=질문|$)',
                ]
                
                matches_found = False
                for pattern in backup_patterns:
                    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
                    if matches:
                        matches_found = True
                        for question, answer in matches:
                            clean_question = self._clean_text(question)
                            clean_answer = self._clean_text(answer)
                            
                            if clean_question and clean_answer:
                                qa_pairs.append({
                                    'question': clean_question,
                                    'answer': clean_answer
                                })
                        break
                
                if not matches_found:
                    print(f"⚠️ QA 파싱 실패 - 텍스트 샘플: {text[:200]}...")
                    # 완전 실패시에도 빈 리스트 반환 (오류 데이터 생성하지 않음)
            
            return qa_pairs
        
        def _clean_text(self, text):
            """텍스트에서 불필요한 번호, 태그, 공백 제거"""
            import re
            if not text:
                return ""
            
            # 앞뒤 공백 제거
            text = text.strip()
            
            # 시작 부분의 번호 제거 (예: "1:", "1.", "1 ")
            text = re.sub(r'^\d+[\.:]\s*', '', text)
            
            # HTML 태그나 특수 태그 제거
            text = re.sub(r'<[^>]+>', '', text)
            
            # </think> 같은 태그 제거
            text = re.sub(r'</?\w+>', '', text)
            
            # 연속된 공백을 하나로
            text = re.sub(r'\s+', ' ', text)
            
            # 다시 앞뒤 공백 제거
            text = text.strip()
            
            return text
    
    return MockResponse(response_text, response_format)


def _random_agent():
    """ API Header의 Agent중 1개를 랜덤으로 선택 """
    return random.choice(AGENT_LIST)

def naver_search(
        query: str,
        api_client_id: str = None,
        api_client_secret: str = None,
        category: Literal["blog", "news", "kin", "encyc", "cafearticle", "webkr"] = "news",
        display: int = 20,
        start: int = 1,
        sort: str = "sim"
) -> json:
    """ Naver 검색 API를 통해 데이터를 수집합니다.
    수집한 데이터는 모델이 generate할 때 RAG로 사용할 수 있습니다.
    Naver 검색 API로 얻는 정보가 부정확할때가 많기 때문에 학습용도로만 이용하면 좋을 것 같습니다.
    자세한 내용은 https://developers.naver.com/docs/serviceapi/search/blog/blog.md를 참고하세요.

    Args:
        query: 검색하려는 쿼리값
        api_client_id: 네이버 검색 API이용을 위한 발급 받은 client_id 값
          - 환경변수는 'NAVER_API_ID'로 설정하세요
        api_client_secret: 네이버 검색 API이용을 위한 발급 받은 client_secret 값
          - 환경변수는 'NAVER_API_SECRET'으로 설정하세요
        category: 검색하려는 카테고리, 아래 카테고리로 검색이 가능합니다
          - blog: 블로그
          - news: 뉴스
          - kin: 지식인
          - encyc: 백과사전
          - cafearticle: 카페 게시글
          - webkr: 웹문서
        display: 검색 결과 수 지정, default = 20
        start: 검색 페이지 값
        sort: 정렬값
          - 'sim': 정확도 순으로 내림차순 정렬
          - 'date': 날짜 순으로 내림차순 정렬

    Returns: API로부터 제공받은 검색 결과 response값

    """

    if not (api_client_id and api_client_secret):
        # NAVER DELVEOPER에서
        api_client_id = load_env("NAVER_API_ID")
        api_client_secret = load_env("NAVER_API_SECRET")
    if not api_client_id or not api_client_secret:
        id_ok = "'NAVER_API_ID'" if not api_client_id else ""
        secret_ok = "'NAVER_API_SECRET'" if not api_client_id else ""
        raise ValueError(f"{id_ok} {secret_ok} Not setted")

    url = f"https://openapi.naver.com/v1/search/{category}.json"
    headers = {"X-Naver-Client-Id": api_client_id, "X-Naver-Client-Secret": api_client_secret}
    query = query.encode("utf8")
    params = {"query": query, "start":start, "display":display, "sort":sort}
    response = requests.get(url, params=params, headers=headers)

    if response.status_code == 200:
        return json.loads(response.content.decode())
    else:
        return response.raise_for_status()

def _completion(user_prompt, system_prompt: str = "", response_format = None, model: str = "exaone-4.0-32b"):
    """ 로컬 LLM으로 Completion을 생성 (기존 GPT API 대신)

    Args:
        user_prompt: User Prompt
        system_prompt: System Prompt
        response_format: 현재는 사용하지 않음 (로컬 LLM에서는 JSON 파싱을 별도로 처리)
        model: 모델명 (현재는 로컬 모델 사용)

    Returns: 로컬 LLM으로부터 제공받은 답변 텍스트
    """
    # 로컬 LLM 사용
    return _local_completion(user_prompt, system_prompt, response_format, model)

def get_request(url: str, headers: dict = None, params: dict = None):
    """ URL로 GET Request 요청

    Args:
        url: GET을 요청할 URL
        headers: 헤더
        params: 추가 파라미터

    Returns: Request의 Response
    """
    agent = {"User-Agent": _random_agent()}
    if headers is not None:
        agent = agent.update(headers)
    response = requests.get(url=url, params=params, headers=agent)
    if response.status_code == 200:
        return response
    else:
        return response.raise_for_status()


def get_public_api_data(serviceKey: str = None, base_url: str = None, params: dict = None, **kwargs):
    """공공데이터에서 Request 함수처리

    Args:
        serviceKey: 발급받은 인증키
        base_url: API의 엔트리포인트 URL
        params: 파라미터

    Returns: 공공데이터 API로부터 응답을 받은 Response
    """
    if not serviceKey:
        serviceKey = load_env(key="PUBLIC_DATA_API_KEY", fname=".env")
    if not params:
        params = {}
    params.update(dict(serviceKey=serviceKey))
    params.update(kwargs)
    response = requests.get(url=base_url, params=params)
    return response

def _mcqa_gpt_completion(
        data: List[Dict] = None,
        task_type: Literal["common", "knowledge", "math"] = None,
        domain_type: Literal["common", "accounting", "market"] = None,
        if_fewshot: bool = True,
        n_datasets: int = 5,
        step: int = None,
        oai_model = "exaone-4.0-32b"
):
    """ MCQA로 전달되는 GPT Completion

    Args:
        data: Dict[List] 형태의 Data
        task_type: 사전 정의된 작업 형태를 선택. system prompt에 task에 따른 프롬프트가 추가.
          - knowledge: 일반적 지식. 예시) ...상세한 답변과 추론 과정을 작성...
          - math: 수학 문제. 예시) ... 수학적으로 계산하는 문제를 작성하고...
          - common: 특정 도메인을 적용하기 어려울 때 포괄적으로 선택. 예시) 금융, 회계, 금융 법률, 그리고 수학적 계산 및 분석을 포함하는...
        domain_type: 생성할 프롬프트의 도메인 형태. 도메인에 따라 페르소나(role) 프롬프트가 추가.
          - accounting: 회계. 예시) 당신은 전문 회계사 역할을 맡아...
          - market: 금융시장. 예시) 당신은 금융 및 주식 분석가 역할을 맡아...
          - common: 특정 도메인을 적용하기 어려울 때 포괄적으로 선택. 예시) 당신은 유능한 금융 전문가입니다...포괄적인 도움을 제공합니다.
        if_fewshot: fewshot을 사용할지 여부
        n_datasets: 1개의 레퍼런스당 생성할 데이터 수
        step: Evolve-Instruct를 모티브로 하여 생성하려는 Instruction의 난이도 설정
        oai_model: GPT 모델명

    Returns: MCQA 형태의 데이터셋 출력
    """
    if not step:
        raise ValueError(f"{step=}\nstep should be defined")
    prompt = PromptTemplates()
    path = PathConfig()

    chunk_index = data['chunk_index']
    if chunk_index == 0:
        n_options = 4
    if chunk_index == 1:
        n_options = 5
    if chunk_index == 2:
        n_options = 6
    if chunk_index == 3:
        n_options = 7
    fewshots_text_name = f"common_{task_type}_mcqa_n_options_{n_options}.txt"
    if domain_type == 'common':
        fewshots_text_name = random.choice([
            f"common_knowledge_mcqa_n_options_{n_options}.txt",
            f"common_math_mcqa_n_options_{n_options}.txt"
            ])

    with open(os.path.join(path.FEWSHOT_DIR, fewshots_text_name), "r") as f:
        fewshots = f.read().split("<SEP>")

    fewshot = fewshots[random.randint(0, len(fewshots) - 1)]
    system_prompt = prompt.system_prompt(
        eval_type="mcqa",
        n_options=n_options,
        example=fewshot if if_fewshot else None,
        task_type=task_type,
        domain_type=domain_type,
        step=step
    )
    user_prompt = prompt.user_prompt(eval_type="mcqa", n_datasets=n_datasets, step=step, data=data)
    print(f"mcqa-system:\n{system_prompt}")
    print(f"mcqa-user:\n{user_prompt}")
    response = _completion(system_prompt=system_prompt, user_prompt=user_prompt, response_format=MCQAOutputFormat, model=oai_model)
    if data.get("table_desc", None):
        question = [q.question + "\n" + data['table_desc'] for q in response.choices[0].message.parsed.questions]
    else:
        question = [q.question for q in response.choices[0].message.parsed.questions]

    return {
        "index": [data["index"] for _ in response.choices[0].message.parsed.questions],
        "title": [data['title'] for _ in response.choices[0].message.parsed.questions],
        "contents": [data['contents'] for _ in response.choices[0].message.parsed.questions],
        "question": question,
        "options": [q.options for q in response.choices[0].message.parsed.questions],
        "reasoning_process": [a.reasoning_process for a in response.choices[0].message.parsed.answers],
        "answer": [a.answer for a in response.choices[0].message.parsed.answers],
        "step": [step for _ in response.choices[0].message.parsed.answers],
        "hallucination": ["" for _ in response.choices[0].message.parsed.answers],
        "value": ["" for _ in response.choices[0].message.parsed.answers],
        "type": ["mcqa"  for _ in response.choices[0].message.parsed.answers],
        "oai_model": [oai_model for _ in response.choices[0].message.parsed.answers],
        "update_at": [datetime.now().strftime("%Y-%m-%d") for _ in response.choices[0].message.parsed.answers]
    }

def _qa_gpt_completion(
        data: List[Dict] = None,
        task_type: Literal["common", "knowledge", "math"] = None,
        domain_type: Literal["common", "accounting", "market", "law", "quant"] = None,
        n_datasets: int = 5,
        step: int = None,
        oai_model = "exaone-4.0-32b"
):
    """ QA로 전달되는 GPT Completion

    Args:
        data: Dict[List] 형태의 Data
        task_type: 사전 정의된 작업 형태를 선택. system prompt에 task에 따른 프롬프트가 추가.
          - knowledge: 일반적 지식. 예시) ...상세한 답변과 추론 과정을 작성...
          - math: 수학 문제. 예시) ... 수학적으로 계산하는 문제를 작성하고...
          - common: 특정 도메인을 적용하기 어려울 때 포괄적으로 선택. 예시) 금융, 회계, 금융 법률, 그리고 수학적 계산 및 분석을 포함하는..
        domain_type: 생성할 프롬프트의 도메인 형태. 도메인에 따라 페르소나(role) 프롬프트가 추가.
          - accounting: 회계. 예시) 당신은 전문 회계사 역할을 맡아...
          - market: 금융시장. 예시) 당신은 금융 및 주식 분석가 역할을 맡아...
          - law: 법률. 예시) 당신은 금융 법률 전문가 역할을 맡아...
          - quant: 금융학, 금융공학. 예시) 당신은 금융학자 역할을 맡아...
          - common: 특정 도메인을 적용하기 어려울 때 포괄적으로 선택. 예시) 당신은 유능한 금융 전문가입니다...포괄적인 도움을 제공합니다.
        if_fewshot: fewshot을 사용할지 여부
        n_datasets: 1개의 레퍼런스당 생성할 데이터 수
        step: Evolve-Instruct를 모티브로 하여 생성하려는 Instruction의 난이도 설정
        oai_model: GPT 모델명

    Returns: QA 형태의 데이터셋 출력
    """
    if not step:
        raise ValueError(f"{step=}\nstep should be defined and one of (1, 2)")
    prompt = PromptTemplates()

    system_prompt = prompt.system_prompt(eval_type="qa", n_options=None, task_type=task_type, domain_type=domain_type, step=step)
    if step == 1:
        user_prompt = prompt.user_prompt(eval_type="qa", n_datasets=n_datasets, step=step, data=data)
    if step == 2:
        user_prompt = prompt.user_prompt(eval_type="qa", step=step, data=data)

    response = _completion(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_format=QAOutputFormat if step == 1 else QAAnswerModel,
        model=oai_model
    )
    print(f"qa-system:\n{system_prompt}")
    print(f"qa-user:\n{user_prompt}")
    if step == 1:
        reference_index = "_".join(data['index'].split("_")[:-1])
        return {
            "reference_index": [reference_index for _ in response.choices[0].message.parsed.questions],
            "index": [data["index"] for _ in response.choices[0].message.parsed.questions],
            "title": [data['title'] for _ in response.choices[0].message.parsed.questions],
            "contents": [data['contents'] for _ in response.choices[0].message.parsed.questions],
            "question": [q.question for q in response.choices[0].message.parsed.questions],
            "old_answer": [a.answer for a in response.choices[0].message.parsed.answers],
            "answer": ["" for _ in response.choices[0].message.parsed.answers],
            "preference": ["" for _ in response.choices[0].message.parsed.answers],
            "preference_desc":  ["" for _ in response.choices[0].message.parsed.answers],
            "value": ["" for _ in response.choices[0].message.parsed.answers],
            "type": ["qa" for _ in response.choices[0].message.parsed.answers],
            "oai_model": [oai_model for _ in response.choices[0].message.parsed.answers],
            "update_at": [datetime.now().strftime("%Y-%m-%d") for _ in response.choices[0].message.parsed.answers]

        }

    if step == 2:
        return {
            "reference_index": "_".join(data['index'].split("_")[:-1]),
            "index": data["index"],
            "title": data['title'],
            "contents": data['contents'],
            "question": data['question'],
            "old_answer": data['old_answer'],
            "answer": response.choices[0].message.parsed.answer,
            "preference": "",
            "preference_desc": "",
            "value": "",
            "type": 'qa',
            "oai_model": oai_model,
            "update_at": datetime.now().strftime("%Y-%m-%d")
        }

def _hallucination_gpt_completion(
        data: List[Dict] = None,
        oai_model="exaone-4.0-32b"
):
    """ 생성한 데이터셋이 Hallucination인지 판단. 입력되는 data는 _mcqa_gpt_completion 또는 _qa_gpt_completion의 형태여야함

    Args:
        data: hallucination 판단을 받기 위한 데이터
        oai_model: GPT 모델명

    Returns: hallucination 판단이 포함된 data

    """

    prompt = PromptTemplates()

    system_prompt = prompt.system_prompt(eval_type='hallucination')
    user_prompt = prompt.user_prompt(eval_type="hallucination", data=data)
    response = _completion(system_prompt=system_prompt, user_prompt=user_prompt, response_format=HallucinationOutputFormat, model=oai_model)
    hallucination = response.choices[0].message.parsed.hallucination.lower()
    print(f"hallu-system:\n{system_prompt}")
    print(f"hallu-user:\n{user_prompt}")

    data.update({
        "hallucination": hallucination,
        "oai_model": oai_model,
        "update_at": datetime.now().strftime("%Y-%m-%d")
    })

    return data

def _preference_gpt_completion(
        data: List[Dict] = None,
        oai_model="exaone-4.0-32b"
):
    """ 생성한 데이터셋의 답변 2개중 Preference를 판단. _qa_gpt_completion에서 2번째 답변까지 생성된 상태여야함

    Args:
        data: preference 판단을 받기 위한 데이터
        oai_model: GPT 모델명

    Returns: preference 판단이 포함된 data

    """
    prompt = PromptTemplates()

    system_prompt = prompt.system_prompt(eval_type='preference')
    user_prompt = prompt.user_prompt(eval_type="preference", data=data)
    response = _completion(system_prompt=system_prompt, user_prompt=user_prompt, response_format=PreferenceOutputFormat, model=oai_model)
    choice = response.choices[0].message.parsed.choice[0].upper()
    print(f"pref-system:\n{system_prompt}")
    print(f"pref-user:\n{user_prompt}")

    data.update({
        "preference": choice,
        "preference_desc": response.choices[0].message.parsed.preference_desc,
        "oai_model": oai_model,
        "update_at": datetime.now().strftime("%Y-%m-%d")
    })
    return data

def _value_gpt_completion(
        data: List[Dict] = None,
        oai_model="exaone-4.0-32b"
):
    """ 생성한 데이터셋의 답변을 Fineweb-edu 기반 프롬프트로 교육적 가치를 0~5점사이의 점수로 판단

    Args:
        data: value 판단을 받기 위한 데이터
        oai_model: GPT 모델명

    Returns: value 측정이 포함된 data

    """
    prompt = PromptTemplates()

    system_prompt = prompt.system_prompt(eval_type='value')
    user_prompt = prompt.user_prompt(eval_type="value", data=data)
    response = _completion(system_prompt=system_prompt, user_prompt=user_prompt, response_format=ValueOutputModel, model=oai_model)
    print(f"value-system:\n{system_prompt}")
    print(f"value-user:\n{user_prompt}")
    data.update({
        "value": response.choices[0].message.parsed.value,
        "oai_model": oai_model,
        "update_at": datetime.now().strftime("%Y-%m-%d")
    })
    return data

def _classification_gpt_completion(
        data: List[Dict] = None,
        oai_model="exaone-4.0-32b"
):
    """ 데이터가 어떤 task_type, domain_type인지 분류를 요청
    예시) knowledge_accounting, knowledge_market, quant, law...

    Args:
        data: 분류하려는 데이터
        oai_model: gpt 모델명

    Returns: 분류가 완료된 데이터

    """
    prompt = PromptTemplates()

    system_prompt = prompt.system_prompt(eval_type='classification')
    user_prompt = prompt.user_prompt(eval_type="classification", data=data)
    response = _completion(system_prompt=system_prompt, user_prompt=user_prompt, response_format=ClassificationOutputModel, model=oai_model)
    print(f"cls-system:\n{system_prompt}")
    print(f"cls-user:\n{user_prompt}")
    data.update({
        "classification": response.choices[0].message.parsed.classification.lower(),
        "oai_model": oai_model,
        "update_at": datetime.now().strftime("%Y-%m-%d")
    })

    return data

def completion(
        data: List[Dict],
        eval_type: Literal['mcqa', 'qa', 'hallucination', 'preference', 'classification'] = None,
        task_type: Literal["common", "knowledge", "math"] = None,
        domain_type: Literal["common", "accounting", "market", "law", "academy"] = None,
        n_datasets: int = 5,
        step: int = None,
        oai_model = "exaone-4.0-32b",
        **kwargs
):
    """ 모든 Completion을 통합하는 API

    Args:
        data: GPT로 생성하려는 소스 데이터
        eval_type: GPT로부터 평가받을 형태를 입력. 형태에 따라 사전 입력된 System Prompt의 jinja2 템플릿 호출
          - mcqa: MCQA Instruction 생성 작업
          - qa: QA Instruction 생성 작업
          - hallucination: Hallucination 여부 평가
          - preference: 첫번째, 두번째 Instruction중 Preference 평가
          - value: Fineweb-edu 기반 교육적 가치 측정
        task_type: 사전 정의된 작업 형태를 선택. system prompt에 task에 따른 프롬프트가 추가.
          - knowledge: 일반적 지식. 예시) ...상세한 답변과 추론 과정을 작성...
          - math: 수학 문제. 예시) ... 수학적으로 계산하는 문제를 작성하고...
          - common: 특정 도메인을 적용하기 어려울 때 포괄적으로 선택. 예시) 금융, 회계, 금융 법률, 그리고 수학적 계산 및 분석을 포함하는...
        domain_type: 생성할 프롬프트의 도메인 형태. 도메인에 따라 페르소나(role) 프롬프트가 추가.
          - accounting: 회계. 예시) 당신은 전문 회계사 역할을 맡아...
          - market: 금융시장. 예시) 당신은 금융 및 주식 분석가 역할을 맡아...
          - law: 법률. 예시) 당신은 금융 법률 전문가 역할을 맡아...
          - quant: 금융학, 금융공학. 예시) 당신은 금융학자 역할을 맡아...
          - common: 특정 도메인을 적용하기 어려울 때 포괄적으로 선택. 예시) 당신은 유능한 금융 전문가입니다...포괄적인 도움을 제공합니다.
        n_datasets: 1개의 레퍼런스로부터 생성하려는 QA 및 MCQA의 갯수
        step: Evolve-Instruct를 모티브로 하여 생성하려는 Instruction의 난이도 설정
        oai_model: GPT 모델명
        **kwargs: 기타 파라미터

    Returns:
        GPT가 생성한 답변
    """

    if eval_type == 'mcqa':
        return _mcqa_gpt_completion(
            data=data,
            task_type=task_type,
            domain_type=domain_type,
            if_fewshot=kwargs['if_fewshot'],
            n_datasets=n_datasets,
            step=step,
            oai_model=oai_model
        )

    if eval_type == 'qa':
        return _qa_gpt_completion(
            data=data,
            task_type=task_type,
            domain_type=domain_type,
            n_datasets=n_datasets,
            step=step,
            oai_model=oai_model
        )

    if eval_type == 'hallucination':
        return _hallucination_gpt_completion(
            data=data,
            oai_model=oai_model
        )

    if eval_type == 'preference':
        return _preference_gpt_completion(
            data=data,
            oai_model=oai_model
        )
    if eval_type == 'value':
        return _value_gpt_completion(
            data=data,
            oai_model=oai_model
        )

    if eval_type == 'classification':
        return _classification_gpt_completion(
            data=data,
            oai_model=oai_model
        )

def calculate_tokens(string: str, encoding_name: str = "o200k_base") -> int:
    """ tiktoken 기반으로 토큰 수를 계산 (tiktoken이 없으면 근사치 계산)

    Args:
        string: 인코딩하려는 텍스트값
        encoding_name: o200k_base: gpt-4o, gpt-4o-mini

    Returns: 토큰 수

    """
    if OPENAI_AVAILABLE and tiktoken:
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens
    else:
        # tiktoken이 없을 때 근사치 계산 (1토큰 ≈ 4글자)
        return len(string) // 4

