from openai import OpenAI
import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI()

st.title("AI chatbot")
user_input = st.text_input("question")

if st.button("ask"):
    with st.spinner("thinking..."):
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": user_input,
                }
            ],
            max_tokens=200,
        )

        generated_text = completion.choices[0].message.content
        st.text(generated_text)


    # completion = client.chat.completions.create(
    #     model="gpt-3.5-turbo",
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": user_input,
    #         }
    #     ],
    #     max_tokens=200,
    # )

    # generated_text = completion.choices[0].message.content
    # print(generated_text)
