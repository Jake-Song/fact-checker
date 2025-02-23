import streamlit as st
from google import genai
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
import os
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
model_id = "gemini-2.0-flash"

google_search_tool = Tool(
    google_search = GoogleSearch()
)

def generate_text(prompt):
    response = client.models.generate_content(
        model=model_id,
        contents=prompt,
        config=GenerateContentConfig(
            tools=[google_search_tool],
            response_modalities=["TEXT"],
        )
    )
    return response

st.title("LLM Serving App")
st.write("Enter a prompt and generate text using a pre-trained LLM.")

prompt = st.text_area("Enter your prompt:", value="")

if st.button("Generate Text"):
    with st.spinner("Generating..."):
        response = generate_text(prompt)
        for each in response.candidates[0].content.parts:
            st.subheader("Generated Text")
            st.write(each.text)
        if response.candidates[0].grounding_metadata.search_entry_point is not None:
            st.markdown(response.candidates[0].grounding_metadata.search_entry_point.rendered_content, unsafe_allow_html=True)
        else:
            st.write("No search results")