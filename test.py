from reward_sampling import RewardSampling
import torch
import json
from tqdm import tqdm

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
    device=device,
)

# Load prompts from HH.json
with open("HH.json", "r") as f:
    prompts = json.load(f)

# Create output file
output_file = "generated_responses.jsonl"

with open(output_file, "w") as f:
    i = 0
    for prompt_dict in tqdm(prompts):
        # print(f"prompt: {prompt.keys()} {prompt['prompt']}")
        prompt = prompt_dict["prompt"]
        # Generate response using rs.rs_generate
        output, (reward, num_llm_call, num_rm_call) = rs.rs_generate(
            [prompt],
            max_new_token=128,
            # entropy_threshold=3.0,
            # reward_threshold=8.5,
            # alpha=0.5,
            # beta=0.7,
        )

        # Save results
        result = {
            "prompt": prompt,
            "response": output[0],
            "reward": reward,
            "num_llm_call": num_llm_call,
            "num_rm_call": num_rm_call,
        }

        json.dump(result, f)
        f.write("\n")
        f.flush()
        i += 1
        if i > 10:
            break

print("Generation complete! Results saved to", output_file)
