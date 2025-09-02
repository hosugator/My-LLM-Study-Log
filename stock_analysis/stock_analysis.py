from dotenv import load_dotenv
import os
import openai
import streamlit as st


load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

select_model = "gpt-3.5-turbo"
user_input = st.text_input("question")

completion = openai.ChatCompletion.create(
    model=select_model, messages=[
        {"role": "user", 
         "content": user_input}]
)

print(completion.choices[0].message.content)
