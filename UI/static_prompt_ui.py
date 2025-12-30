from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.header("Research Tool")

user_input = st.text_area("Enter your research query here:")

if st.button("Submit") and user_input.strip():
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        task="text-generation",
        max_new_tokens=200,
        do_sample=False,
        repetition_penalty=1.03,
        temperature=0.7,
    )

    chat = ChatHuggingFace(llm=llm)
    result = chat.invoke(user_input)

    st.subheader("Response:")
    st.write(result.content)
