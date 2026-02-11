
import streamlit as st
import requests





API_URL = "http://127.0.0.1:8000/predict"


st.set_page_config(page_title="Next Word Predictor", page_icon="🤖")

st.title("Next Word Prediction")
st.write("Type a sentence and let the model guess the next word.")

text = st.text_input("Enter text:")


if st.button("Predict"):

    if text.strip() == "":
        st.warning("Please enter some text")
    else:
        try:
            response = requests.post(API_URL, params={"text": text})
            result = response.json()["next_word"]

            st.success(f"Predicted next word: **{result}**")

        except:
            st.error("Backend not running. Start FastAPI server first.")
