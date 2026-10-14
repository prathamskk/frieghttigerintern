import os
import sys
import time
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

print("=== Starting Model Pre-downloading for Offline Usage ===")

# 1. Download Sentence Transformer Embedding Model
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
print(f"\n[1/3] Downloading Embedding Model: {embedding_model_name}...")
t0 = time.time()
embedder = SentenceTransformer(embedding_model_name)
test_embed = embedder.encode(["Freight shipping delay on Chennai-Bangalore highway due to rain"])
print(f"Embedding Model downloaded and verified in {time.time() - t0:.2f}s! Vector shape: {test_embed.shape}")

# 2. Download Qwen2.5-0.5B-Instruct (Lightweight, ultra-fast CPU fallback)
llm_05b_name = "Qwen/Qwen2.5-0.5B-Instruct"
print(f"\n[2/3] Downloading LLM: {llm_05b_name} (Fast ~900MB)...")
t0 = time.time()
tokenizer_05b = AutoTokenizer.from_pretrained(llm_05b_name)
model_05b = AutoModelForCausalLM.from_pretrained(
    llm_05b_name,
    torch_dtype=torch.float32,
    device_map="cpu",
    low_cpu_mem_usage=True
)
print(f"{llm_05b_name} downloaded in {time.time() - t0:.2f}s!")

# Quick test of 0.5B
messages = [
    {"role": "system", "content": "You are a freight cost analyst."},
    {"role": "user", "content": "Explain briefly why a 10% diesel price increase affects long-haul shipping."}
]
text = tokenizer_05b.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer_05b([text], return_tensors="pt")
outputs = model_05b.generate(**inputs, max_new_tokens=40, do_sample=False)
response = tokenizer_05b.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
print(f"Verification output: {response.strip()}")

# 3. Download Qwen2.5-1.5B-Instruct (Primary high-accuracy reasoning model)
llm_15b_name = "Qwen/Qwen2.5-1.5B-Instruct"
print(f"\n[3/3] Downloading LLM: {llm_15b_name} (High Accuracy ~3GB)...")
t0 = time.time()
tokenizer_15b = AutoTokenizer.from_pretrained(llm_15b_name)
model_15b = AutoModelForCausalLM.from_pretrained(
    llm_15b_name,
    torch_dtype=torch.float32,
    device_map="cpu",
    low_cpu_mem_usage=True
)
print(f"{llm_15b_name} downloaded in {time.time() - t0:.2f}s!")

# Quick test of 1.5B
text15 = tokenizer_15b.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs15 = tokenizer_15b([text15], return_tensors="pt")
outputs15 = model_15b.generate(**inputs15, max_new_tokens=40, do_sample=False)
response15 = tokenizer_15b.decode(outputs15[0][len(inputs15.input_ids[0]):], skip_special_tokens=True)
print(f"Verification output: {response15.strip()}")

print("\n=== ALL MODELS SUCCESSFULLY DOWNLOADED AND VERIFIED OFFLINE READY! ===")
