from reward_sampling import RewardSampling
import torch
import json
from tqdm import tqdm
import time
from datetime import datetime

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# rs = RewardSampling(access_token=None, llm_dir='argsearch/llama-7b-sft-float32', rm_dir='argsearch/llama-7b-rm-float32')
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.cuda.set_device(device)

rs = RewardSampling(
    access_token=None,
    # llm_dir="meta-llama/Llama-3.2-1B-Instruct",
    # rm_dir="/home/richardfan/ICML2025_RGTG/ICML_llama_1B_RMv1_TLDR_5e6",
    llm_dir="lomahony/eleuther-pythia2.8b-hh-sft",
    rm_dir="/home/richardfan/ICML2025_RGTG/ICML_pythia_3b_RMv1_HH_5e6",
    device=device,
)

# Load prompts from HH.json
with open("HH.json", "r") as f:
    prompts = json.load(f)

# Create output file
output_file = "generated_responses.jsonl"
batch_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")  # Batch ID using timestamp

start_time = time.time()  # Start timer for this prompt


with open(output_file, "w") as f:
    i = 0
    for prompt_dict in tqdm(prompts):
        prompt = prompt_dict["prompt"]

        output, (reward, num_llm_call, num_rm_call) = rs.rs_generate(
            [prompt],
            max_new_token=128,
            # entropy_threshold=3.0,
            # reward_threshold=8.5,
            # alpha=0.5,
            # beta=0.7,
        )

        elapsed_time = time.time() - start_time  # Calculate elapsed time

        # Save results
        result = {
            "batch_id": batch_start_time,
            "prompt_index": i,
            "prompt": prompt,
            "response": output[0],
            "reward": reward,
            "num_llm_call": num_llm_call,
            "num_rm_call": num_rm_call,
            "elapsed_time": elapsed_time,
        }

        json.dump(result, f)
        f.write("\n")
        f.flush()
        i += 1
        if i > 10:
            break

print("Generation complete! Results saved to", output_file)
