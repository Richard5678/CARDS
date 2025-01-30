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

# root = "/home/richardfan"
root = "/scratch/ssd004/scratch/rofan"
diversity_num = 10
# dataset = "HH"
# dataset = "TLDR"
dataset = "v2_TLDR"
# reward_threshold = 8.5
# reward_threshold = 6.375
# reward_threshold = 4.25
# reward_threshold = 3.1875
# reward_threshold = 2.125

reward_threshold = 2.08 / 2 if dataset in ["TLDR", "v2_TLDR"] else 1.73 / 2
# reward_threshold = 2.08 if dataset in ["TLDR", "v2_TLDR"] else 1.73


if dataset == "HH":
    llm_dir = "lomahony/eleuther-pythia2.8b-hh-sft"
    rm_dir = f"{root}/ICML2025_RGTG/ICML_pythia_3b_RMv1_HH_5e6"
elif dataset in ["TLDR", "v2_TLDR"]:
    llm_dir = "meta-llama/Llama-3.2-1B-Instruct"
    rm_dir = f"{root}/ICML2025_RGTG/ICML_llama_1B_RMv1_TLDR_5e6"

rs = RewardSampling(
    access_token=None,
    llm_dir=llm_dir,
    rm_dir=rm_dir,
    cache_dir=f"{root}/cache",
    device=device,
)

# Load prompts from HH.json
with open(f"{root}/ICML2025_RGTG/{dataset}.json", "r") as f:
    prompts = json.load(f)

# Create output file
batch_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")  # Batch ID using timestamp
# batch_start_time = "20250128_232417"
output_file = f"generated_responses_{dataset}_{reward_threshold}_{batch_start_time}_{diversity_num}.jsonl"

# start_time = time.time()  # Start timer for this prompt
# Get the last elapsed time from the output file if it exists
prev_time = 0
if os.path.exists(f"{root}/{output_file}"):
    with open(f"{root}/{output_file}", "r") as f:
        # Read all lines and get the last one
        lines = f.readlines()
        if lines:
            last_record = json.loads(lines[-1])
            prev_time = last_record["elapsed_time"]

start_time = time.time()
    
print(f"Prev time: {prev_time}")


with open(f"{root}/{output_file}", "a") as f:
    for i, prompt_dict in enumerate(tqdm(prompts)):
        # if i < 54:
        #     continue
        prompt = prompt_dict["prompt"]

        for j in range(diversity_num):  
            output, (reward, num_llm_call, num_rm_call) = rs.rs_generate(
                [prompt],
                max_new_token=128 if dataset == "HH" else 64,
                # entropy_threshold=3.0,
                reward_threshold=reward_threshold,
                topk=10,
            # alpha=0.5,
            # beta=0.7,
            )

            # Remove the prompt from the output
            generated_text = output[0].replace(prompt, "").strip()

            elapsed_time = prev_time + time.time() - start_time  # Calculate elapsed time

            # Save results
            result = {
                "batch_id": batch_start_time,
                "prompt_index": i,
                "prompt": prompt,
                "response": generated_text,  # Using the cleaned output
                "reward": reward,
                "reward_threshold": reward_threshold,
                "num_llm_call": num_llm_call,
                "num_rm_call": num_rm_call,
                "elapsed_time": elapsed_time,
            }

            json.dump(result, f)
            f.write("\n")
            f.flush()

print("Generation complete! Results saved to", output_file)
