import streamlit as st
import os
from openai import OpenAI
from tavily import TavilyClient
import json
from dotenv import load_dotenv
load_dotenv()

client = OpenAI()

tavily_client = TavilyClient(api_key=f"{os.getenv('TAVILY_API_KEY')}")

def search_tavily(query):
    response = tavily_client.search(
        query=query
    )
    return response['results']

system_prompt = """
You are a helpful assistant that can fact check with credible sources from the web. 
You need to check if the claim is true or false based on the sources.
You add the link of the source to the answer for users to check the credibility of the answer.
"""

def generate_text(user_prompt):
    tools = [{
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for collecting credible sources.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to search the web for credible sources."
                    }
                },
                "required": [
                    "query"
                ],
                "additionalProperties": False
            },
            "strict": True
        }
    }]

    messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
    ]

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools
    )
    if completion.choices[0].message.tool_calls is None:
        return completion.choices[0].message.content
    else:
        tool_call = completion.choices[0].message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)

        result = search_tavily(args["query"])
        
        messages.append(completion.choices[0].message)  # append model's function call message
        messages.append({                               # append result message
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": str(result)
        })

        completion_2 = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
        )

        return completion_2.choices[0].message.content

st.title("Fact Checker")
st.write("Enter a claim and check if it is true or false with credible sources from the web.")

claim = st.text_area("Enter your claim:", value="")

if st.button("Check Fact"):
    with st.spinner("Checking..."):
        response = generate_text(claim)
        st.subheader("Generated Text")
        st.write(response)
  