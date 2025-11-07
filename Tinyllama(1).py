#!/usr/bin/env python
# coding: utf-8

# In[ ]:


model_name = "AinzDerrick/tinyllama-demo"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

