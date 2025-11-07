#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

st.title("🦙 TinyLLaMA Chatbot")
st.write("A lightweight LLM demo you can run on Streamlit Cloud or locally.")

@st.cache_resource
def load_model():
    # Point directly to the subfolder containing the model files
    model_name = "AinzDerrick/tinyllama-demo/TinyLLaMAModel"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True
    )
    return tokenizer, model

tokenizer, model = load_model()

prompt = st.text_area("💭 Your message:", "Hello, TinyLLaMA!")

if st.button("Generate Response"):
    with st.spinner("Thinking..."):
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.write("### 🤖 TinyLLaMA says:")
    st.write(response)

