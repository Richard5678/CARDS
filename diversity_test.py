import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import torch
import time
import pandas as pd
import matplotlib.pyplot as plt
import os
from datasets import load_dataset, load_from_disk, Dataset
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json
import sys
from transformers import set_seed


tqdm.pandas()
# =========================================================================================
# =================================== Load the models ===================================
device1 = "cuda:0"
device2 = "cuda:1"

text_output = []

# base model
base_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", cache_dir="/h/ruotian/ruotian")
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", cache_dir="/h/ruotian/ruotian", torch_dtype=torch.bfloat16).to(device1)
base_model.eval()

# v1 reward models
v1_reward_tokenizer = AutoTokenizer.from_pretrained("/h/ruotian/ruotian/ICML_llama_1B_RMv1_TLDR_noSFT_5e6", padding_side='left', cache_dir="/h/ruotian/ruotian")
v1_reward_model = AutoModelForSequenceClassification.from_pretrained("/h/ruotian/ruotian/ICML_llama_1B_RMv1_TLDR_noSFT_5e6", num_labels=1, torch_dtype=torch.bfloat16, cache_dir="/h/ruotian/ruotian").to(device2)
v1_reward_model.eval()

# v2 reward models
v2_reward_tokenizer = AutoTokenizer.from_pretrained("/h/ruotian/ruotian/ICML_llama_1B_RMv2_TLDR_noSFT_5e6_iter5", padding_side='left', cache_dir="/h/ruotian/ruotian")
v2_reward_model = AutoModelForCausalLM.from_pretrained("/h/ruotian/ruotian/ICML_llama_1B_RMv2_TLDR_noSFT_5e6_iter5", torch_dtype=torch.bfloat16, cache_dir="/h/ruotian/ruotian").to(device2)
v2_reward_model.eval()


# =========================================================================================

def evaluate_reward_v1(Q, A, reward_tokenizer, reward_model):
    inputs = reward_tokenizer(Q, A, return_tensors='pt').to(device2)
    external_reward = reward_model(**inputs).logits[0].cpu().detach().item()
    return external_reward

# ======================================================================================================
# ============================================== RGNS DECODING =========================================
# mode1: 1 - greedy, 2 - optimal policy

def RGTG_decoding_v2(llm_model=None, llm_tokenizer=None, reward_model=None, reward_tokenizer=None, 
                topk=10, prompt=None, max_generation_length=128, mode=2, w=2.0):
    
    tokenizer_output = llm_tokenizer(prompt, return_tensors='pt').to(device1)
    input_ids = tokenizer_output.input_ids
    
    # use sequence to keep the entire generation
    sequence = torch.tensor([[]],dtype=torch.int64).to(device1)

    for t in range(0, max_generation_length):
        if t == 0:
            llm_output = llm_model(input_ids=input_ids)
            RM_output = reward_model(input_ids=input_ids.to(device2))
        else:
            llm_output = llm_model(input_ids=torch.cat((input_ids, sequence), dim=-1))
            RM_output = reward_model(torch.cat((input_ids, sequence), dim=-1).to(device2))
        
        LLM_logits = llm_output.logits[0, -1]
        RM_logits = RM_output.logits[0, -1].to(device1)
        
        RG_logit = LLM_logits + w * RM_logits
        
        if mode == 1:
            sampled_token_id = torch.topk(RG_logit, 1).indices[0].reshape(1, 1)
        elif mode == 2:
            topk_values, topk_indices = torch.topk(RG_logit, topk, dim=-1)
            sampled_index = torch.distributions.categorical.Categorical(logits=topk_values).sample().item()
            sampled_token_id = topk_indices[sampled_index].reshape(1, 1)
            
        sequence = torch.cat((sequence, sampled_token_id), dim=-1)
        
        if sequence[0][-1].item() == llm_tokenizer.eos_token_id:
            print(f"EOS BREAK: {t}")
            break
    
    generation = llm_tokenizer.decode(sequence[0])
    return {"sequence": generation}


def test_RGTG(prompt=None, topk=10, max_generation_length=128, mode=2, w=2.0):
    v2_score = 0
    w = 1.0
    
    for j in range(0, 10):
        torch.manual_seed(41 + j)
        set_seed(42 + j)
        # v2
        v2_output = RGTG_decoding_v2(llm_model=base_model, llm_tokenizer=base_tokenizer, 
                    reward_model=v2_reward_model, reward_tokenizer=v2_reward_tokenizer,
                    topk=topk, prompt=prompt, max_generation_length=max_generation_length, mode=mode, w=w)
        text_output.append({"prompt": prompt, "result": v2_output['sequence']})
    
    return 0

def test_all(sample_size=50, seed=42, topk=10, max_generation_length=128):
    
    tldr_dataset = load_dataset("CarperAI/openai_summarize_tldr")
    test_data = tldr_dataset["test"]
    test_data = test_data.shuffle(seed=seed)
    
    for i in tqdm(range(0, sample_size)):
        prompt = "Summarize: " + test_data[i]['prompt']
        print(f"Prompt:{prompt}\n")
        test_RGTG(prompt, topk, max_generation_length)
    
    with open("Results/v2_TLDR_diversity.json", "w") as outfile:
        outfile.truncate()
        json.dump(text_output, outfile, ensure_ascii=False)

if __name__ == "__main__":
    test_main(sample_size=10, seed=42, topk=10, max_generation_length=64)

