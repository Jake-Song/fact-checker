import streamlit as st
from agent import OpenAIProvider, Agent, Tool, search_tavily, search_tool

llm = OpenAIProvider()
tool = Tool(search_tool, search_tavily)    

search_keyword_agent = Agent(
    llm=llm, 
    system_prompt="""
        사용자가 요청한 내용에 대한 검색 키워드 생성 어시스턴트입니다.
        너의 임무는 
        1. 요청한 내용에 대한 주장을 세분화 
        2. 각 주장에 대한 검색 키워드를 생성
        최적의 검색 결과를 얻기 위한 키워드를 생성해주세요.
        출력 형식은 다음과 같습니다.
        예시:
        지구는 평평하다 : 지구 평평
        미국은 중국을 넘어서 세계 최대 경제 강국이다 : 미국 중국 경제
    """    
)

search_agent = Agent(
    llm=llm, 
    system_prompt="""
        검색 키워드에 따라 도구를 사용해 인터넷 검색을 해주는 도우미입니다.
        출력 형식은 다음과 같습니다.
        예시: 
        지구는 평평하다 - 제목: 지구는 평평하다, 내용: 지구는 평평하다, 링크: https://www.google.com
        미국은 중국을 넘어서 세계 최대 경제 강국이다 - 제목: 미국은 중국을 넘어서 세계 최대 경제 강국이다, 내용: 미국은 중국을 넘어서 세계 최대 경제 강국이다, 링크: https://www.google.com
    """,
    tool=tool
)

verdict_agent = Agent(
    llm=llm, 
    system_prompt="""
        주어진 내용에 따라 사실인지 아닌 지 판단하고 그 근거를 내용에 기반해 제시해주는 도우미입니다. 링크도 포함해서 제시해주세요.
    """,
)

def fact_check(claim):
    total_tokens = 0
    search_keyword, tokens = search_keyword_agent.generate(claim)
    print(f"search_keyword: {search_keyword}, \n\ntokens: {tokens}")
    total_tokens += tokens

    search_result, tokens = search_agent.generate(search_keyword)
    print(f"search_result: {search_result}, \n\ntokens: {tokens}")
    total_tokens += tokens

    verdict, tokens = verdict_agent.generate(search_result)
    print(f"verdict: {verdict}, \n\ntokens: {tokens}")
    total_tokens += tokens

    print(f"\n\ntotal tokens: {total_tokens}")
    return search_keyword, search_result, verdict, total_tokens
    
st.title("Fact Checker")
st.write("Enter a claim and check if it is true or false with credible sources from the web.")

claim = st.text_area("Enter your claim:", value="")

if st.button("Check Fact"):
    with st.spinner("Checking..."):
        search_keyword, search_result, verdict, total_tokens = fact_check(claim)
        st.subheader("1. Search Keyword:")
        st.write(search_keyword)
        st.subheader("2. Search Result:")
        st.write(search_result)
        st.subheader("3. Verdict:")
        st.write(verdict)
        st.subheader("Total tokens:")
        st.write(total_tokens)
  