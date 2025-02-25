# 🕵️ Fact Checker Agent
팩트 체크를 위한 에이전트 앱입니다.
3개의 에이전트로 구성되어 있습니다. 
- 검색 키워드 생성 에이전트(search_keyword_agent)
- 검색 결과 에이전트(search_agent)
- 팩트 체크 에이전트(verdict_agent)

## Quickstart
1. uv 설치
설치가 되어 있지 않다면 설치 방법을 참고하세요.[`uv` installed](https://docs.astral.sh/uv/)

2. 의존성 설치
```bash
uv sync
```

3. 환경 변수 설정
```bash
cp .env.example .env
```
.env 파일을 생성하고 API 키를 입력해주세요.

4. 서버 실행
```bash
streamlit run app.py
```


