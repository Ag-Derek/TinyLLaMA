#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import requests
import os

st.title("🐪 TinyLLaMA Chatbot (API Version)")
st.write("Uses the updated Hugging Face Router API instead of local model loading.")

# Get Hugging Face API key from environment variable
HF_API_KEY = os.getenv("HF_API_KEY")

if HF_API_KEY is None:
    st.warning("⚠️ Please set your Hugging Face API key as an environment variable named 'HF_API_KEY'.")
else:
    # ✅ NEW endpoint — replaces the deprecated one
    API_URL = "https://router.huggingface.co/hf-inference/models/TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    headers = {"Authorization": f"Bearer {HF_API_KEY}"}

    def query(payload):
        """Send prompt to Hugging Face inference router."""
        response = requests.post(API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 503:
            return {"error": "Model is loading. Please wait a few seconds and try again."}
        elif response.status_code == 404:
            return {"error": "Model not found. Check model name or permissions."}
        elif response.status_code == 410:
            return {"error": "Deprecated endpoint used — please confirm router.huggingface.co is active."}
        else:
            return {"error": f"Request failed with status code {response.status_code}: {response.text}"}

    # Text input area
    prompt = st.text_area("💬 Your message:", "Hello TinyLLaMA!")

    # Generate button
    if st.button("Generate"):
        if not prompt.strip():
            st.warning("Please enter a message first.")
        else:
            with st.spinner("Thinking..."):
                output = query({"inputs": prompt})

                # Handle possible outputs
                if isinstance(output, list) and "generated_text" in output[0]:
                    st.write("### 🤖 TinyLLaMA says:")
                    st.write(output[0]["generated_text"])
                elif "error" in output:
                    st.error(output["error"])
                else:
                    st.write(output)
