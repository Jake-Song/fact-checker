import streamlit as st
from agent import OpenAIProvider, Agent, Tool, search_tavily, search_tool

llm = OpenAIProvider()
tool = Tool(search_tool, search_tavily)    

search_keyword_agent = Agent(
    llm=llm, 
    system_prompt="""
        사용자가 요청한 내용에 대한 팩트 체크를 위한 검색 키워드 생성 어시스턴트로서 
        아래 절차에 따라 검색 키워드를 생성해줘. 최적의 검색 결과를 위해 연산자를 사용해도 좋아.
        1. 사용자가 요청한 내용에 대한 팩트 체크를 위한 검색 키워드 생성.
        2. 요청한 내용을 논점에 따라 구분 
        3. 논점을 팩트 체크하기 위한 검색 키워드를 생성
        최적의 검색 결과를 얻기 위한 키워드를 생성해주세요.
        출력 형식은 다음과 같습니다.
        논점 : 검색 키워드
        예시:      
        지구는 평평하다 : 지구 평평        
    """    
)

search_agent = Agent(
    llm=llm, 
    system_prompt="""
        검색 키워드에 따라 도구를 사용해 인터넷 검색을 해주는 도우미입니다.
        출력 형식은 다음과 같습니다.
        예시: 
        지구는 평평하다 - 제목: 지구는 평평하다, 내용: 지구는 평평하다, 링크: https://www.google.com        
    """,
    tool=tool
)

verdict_agent = Agent(
    llm=llm, 
    system_prompt="""
        주어진 내용에 따라 사실인지 아닌 지 판단하고 그 근거를 내용에 기반해 제시해주는 도우미입니다. 링크도 포함해서 제시해주세요.
    """,
)

dev_llm = OpenAIProvider(model="gpt-4o")

dev_prompt = """
    주어진 내용에 따라 사실인지 아닌 지 판단하고 그 근거를 내용에 기반해 제시해줘.
    판단에 따른 출처를 포함해서 제시해줘.
"""

dev_agent = Agent(
    llm=llm, 
    system_prompt=dev_prompt,
    tool=tool
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

def dev_check(claim):
    dev_result, tokens = dev_agent.generate(claim)
    print(f"dev_result: {dev_result}, \n\ntokens: {tokens}")
    
    return dev_result, tokens

st.title("Fact Checker")
st.write("Enter a claim and check if it is true or false with credible sources from the web.")

claim = st.text_area("Enter your claim:", value="")

if st.button("Check Fact"):
    with st.spinner("Checking..."):
        st.subheader("Dev Check Result:")
        dev_result, tokens = dev_check(claim)
        st.write(dev_result)
        st.write(f"Total tokens: {tokens}")
        # search_keyword, search_result, verdict, total_tokens = fact_check(claim)
        # st.subheader("1. Search Keyword:")
        # st.write(search_keyword)
        # st.subheader("2. Search Result:")
        # st.write(search_result)
        # st.subheader("3. Verdict:")
        # st.write(verdict)
        # st.subheader("Total tokens:")
        # st.write(total_tokens)
  