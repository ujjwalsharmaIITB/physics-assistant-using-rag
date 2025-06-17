
import os

os.environ['HF_HOME'] = "hf_cache"

from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
import torch
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"


# Quantization Config - this allows us to load the model into memory and use less memory

# quant_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     # bnb_4bit_use_double_quant=True,
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_quant_type="nf4"
# )
# 

tokenizer = AutoTokenizer.from_pretrained(LLAMA)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(LLAMA, torch_dtype=torch.bfloat16, trust_remote_code=True)

# Define the pipeline for LLM
hf_pipeline = pipeline('text-generation', model=model, tokenizer=tokenizer, max_new_tokens=1024)

langchain_llm = HuggingFacePipeline(pipeline=hf_pipeline)

chat_model = ChatHuggingFace(llm=langchain_llm)

