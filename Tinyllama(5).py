#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import requests
import os

st.title("🦙 TinyLLaMA Chatbot (API Version)")
st.write("Uses the Hugging Face Inference API instead of local model loading.")

# Get your Hugging Face API key from environment variable
HF_API_KEY = os.getenv("HF_API_KEY")

if HF_API_KEY is None:
    st.warning("⚠️ Please set your Hugging Face API key as an environment variable named 'HF_API_KEY'.")
else:
    #  Updated API endpoint
    API_URL = "https://router.huggingface.co/hf-inference/models/AinzDerrick/tinyllama-demo"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Request failed with status code {response.status_code}: {response.text}"}

    prompt = st.text_area("💭 Your message:", "Hello TinyLLaMA!")

    if st.button("Generate"):
        with st.spinner("Thinking..."):
            output = query({"inputs": prompt})

            # Handle possible errors or malformed responses
            if isinstance(output, list) and "generated_text" in output[0]:
                st.write("### 🤖 TinyLLaMA says:")
                st.write(output[0]["generated_text"])
            elif "error" in output:
                st.error(output["error"])
            else:
                st.write(output)
