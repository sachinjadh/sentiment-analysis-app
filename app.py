# app.py
# Install dependencies before running:
# pip install transformers torch streamlit

from transformers import pipeline
import streamlit as st

# Load model
classifier = pipeline("sentiment-analysis")

# UI
st.title("Sentiment Analysis AI App")
st.write("Enter text below to get the sentiment")

user_input = st.text_area("Your text")

if st.button("Predict Sentiment"):
    if user_input.strip():
        result = classifier(user_input)[0]
        st.write(f"**Sentiment:** {result['label']}")
        st.write(f"**Confidence:** {round(result['score'], 2)}")
    else:
        st.write("Please enter some text.")
