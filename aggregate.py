import json
import argparse
import numpy as np
from pathlib import Path


def analyze_batch(jsonl_path: str, batch_id: str, n_samples_per_query: int=1, idx_in_query: int=0):
    """
    Analyze statistics for a specific batch from a JSONL file.

    Args:
        jsonl_path: Path to the JSONL file
        batch_id: Batch ID to analyze

    Returns:
        tuple: (mean reward, stderr reward, total time, num_llm_call_lst, num_rm_call_lst, num_entries)
    """
    rewards = []
    total_time = 0
    num_llm_call_lst = []
    num_rm_call_lst = []
    num_entries = 0
    prev_time = 0
    total_time = 0  

    # Read and process the JSONL file
    with open(jsonl_path, "r") as f:
        for line_idx, line in enumerate(f):
            record = json.loads(line)
            
            # read every n_samples_per_query lines
            if line_idx % n_samples_per_query == idx_in_query and record["batch_id"] == batch_id:
                rewards.append(record["reward"])
                total_time += record["elapsed_time"] - prev_time
                num_llm_call_lst.append(record["num_llm_call"])
                num_rm_call_lst.append(record["num_rm_call"])
                num_entries += 1

            if line_idx % n_samples_per_query == idx_in_query - 1:
                prev_time = record["elapsed_time"]

    if not rewards:
        raise ValueError(f"No records found for batch_id: {batch_id}")

    # Calculate statistics
    mean_reward = np.mean(rewards)
    stderr_reward = np.std(rewards) / np.sqrt(len(rewards))

    return mean_reward, stderr_reward, total_time, num_llm_call_lst, num_rm_call_lst, num_entries


batch_reward_threshold_table = {
    "HH": {
        "8.5": "20250127_163602",
        "4.25": "20250127_174840",
        "2.125": "20250127_175358",
    },
    "TLDR": {
        "8.5": "20250127_175501",
        "4.25": "20250127_120947",
        "2.125": "20250127_162442",
    },
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze batch statistics from JSONL file"
    )
    parser.add_argument(
        "--root",
        nargs="?",
        help="Root directory",
        type=str,
        default="/scratch/ssd004/scratch/rofan",
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        help="Dataset name",
        type=str,
        # default='generated_responses_HH.jsonl'
        default="HH",
        # default="TLDR",
    )
    parser.add_argument(
        "batch_id",
        nargs="?",
        help="Batch ID to analyze",
        type=str,
        default='20250126_124451', # HH, reward_threshold=8.5
        # default='20250127_174840', # HH, reward_threshold=4.25
        # default='20250127_175358', # HH, reward_threshold=2.125
        
        # default="20250127_175501",  # TLDR, reward_threshold=8.5
        # default="20250127_120947", # TLDR, reward_threshold=4.25
        # default="20250127_162442",  # TLDR, reward_threshold=2.125
    )
    parser.add_argument(
        "n_samples_per_query",
        nargs="?",
        help="Number of samples per query",
        type=int,
        default=1,
    )
    
    parser.add_argument(
        "print_info",
        nargs="?",
        help="Print info",
        type=bool,
        default=False,
    )

    args = parser.parse_args()
    
    return args

def main(args):
    dataset_path = f"{args.root}/generated_responses_{args.dataset}.jsonl"
    # print(f"Dataset path: {dataset_path}")

    try:
        mean_reward, stderr_reward, total_time, num_llm_call_lst, num_rm_call_lst, num_entries = (
            analyze_batch(dataset_path, args.batch_id, args.n_samples_per_query)
        )
        # Convert seconds to HH:MM:SS format
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        mean_num_llm_call = np.mean(num_llm_call_lst)
        mean_num_rm_call = np.mean(num_rm_call_lst)
        stderr_num_llm_call = np.std(num_llm_call_lst) / np.sqrt(len(num_llm_call_lst))
        stderr_num_rm_call = np.std(num_rm_call_lst) / np.sqrt(len(num_rm_call_lst))
        # reward_threshold = args.dataset.split("_")[1]

        if args.print_info:
            print(f"File: {args.dataset}")
            print(f"Batch ID: {args.batch_id}")
            print(f"Reward threshold: {args.reward_threshold}")
            print(f"Mean reward: {mean_reward:.4f}")
            print(f"Standard error reward: {stderr_reward:.4f}")
            print(f"Num entries: {num_entries}")
            print(f"Num LLM call: mean +/- stderr: {mean_num_llm_call:.2f} +/- {stderr_num_llm_call:.2f}")
            print(f"Num RM call: mean +/- stderr: {mean_num_rm_call:.2f} +/- {stderr_num_rm_call:.2f}")
            print(f"Total time: {time_str}")
            print()
            
        return mean_reward, stderr_reward, total_time, mean_num_llm_call, mean_num_rm_call, stderr_num_llm_call, stderr_num_rm_call
    except FileNotFoundError:
        print(f"Error: File {args.jsonl_path} not found")
    except ValueError as e:
        print(f"Error: {e}")
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in file")


if __name__ == "__main__":
    args = parse_args()
    
    # args.batch_id = '20250127_214751'
    # args.dataset = 'HH_2.125_20250127_214751'
    # args.reward_threshold = '2.125'
    # args.batch_id = '20250128_131646'
    # args.dataset = 'TLDR_2.08_20250128_131646'
    # args.reward_threshold = '2.08'
    
    # args.batch_id = '20250128_131648'
    # args.dataset = 'HH_1.73_20250128_131648'
    # args.reward_threshold = '1.73'
    # main(args)
    
    mean_reward_lst = []
    stderr_reward_lst = []
    total_time_lst = []
    mean_num_llm_call_lst = []
    mean_num_rm_call_lst = []
    stderr_num_llm_call_lst = []
    stderr_num_rm_call_lst = []
    
    args.batch_id = '20250128_235908'
    args.dataset = 'HH_1.73_20250128_235908_10'
    args.reward_threshold = '1.73'
    args.n_samples_per_query = 10
    for idx in range(args.n_samples_per_query):
        args.idx_in_query = idx
        mean_reward, stderr_reward, total_time, mean_num_llm_call, mean_num_rm_call, stderr_num_llm_call, stderr_num_rm_call = main(args)
        mean_reward_lst.append(mean_reward)
        stderr_reward_lst.append(stderr_reward)
        total_time_lst.append(total_time)
        mean_num_llm_call_lst.append(mean_num_llm_call)
        mean_num_rm_call_lst.append(mean_num_rm_call)
        stderr_num_llm_call_lst.append(stderr_num_llm_call)
        stderr_num_rm_call_lst.append(stderr_num_rm_call)
    
    # args.batch_id = '20250129_102514'
    # args.dataset = 'HH_0.865_20250129_102514_10'
    # args.reward_threshold = '0.865'
    # args.n_samples_per_query = 10
    # for idx in range(args.n_samples_per_query):
    #     args.idx_in_query = idx
    #     mean_reward, stderr_reward, total_time, mean_num_llm_call, mean_num_rm_call, stderr_num_llm_call, stderr_num_rm_call = main(args)
    #     mean_reward_lst.append(mean_reward)
    #     stderr_reward_lst.append(stderr_reward)
    #     total_time_lst.append(total_time)
    #     mean_num_llm_call_lst.append(mean_num_llm_call)
    #     mean_num_rm_call_lst.append(mean_num_rm_call)
    #     stderr_num_llm_call_lst.append(stderr_num_llm_call)
    #     stderr_num_rm_call_lst.append(stderr_num_rm_call)
        
    print(f"Dataset: {args.dataset}")
    print(f"Reward threshold: {args.reward_threshold}")
    print(f"Reward: {np.mean(mean_reward_lst):.4f} +/- {np.mean(stderr_reward_lst):.4f}")
    print(f"Time: {np.mean(total_time_lst):.4f}")
    print(f"Num LLM call: {np.mean(mean_num_llm_call_lst):.2f} +/- {np.mean(stderr_num_llm_call_lst):.2f}")
    print(f"Num RM call: {np.mean(mean_num_rm_call_lst):.2f} +/- {np.mean(stderr_num_rm_call_lst):.2f}")
    
    # dataset_batch_id_table = {
    #     "HH": {
    #         # "8.5": "20250127_163602",
    #         "8.5": "20250126_124451",
    #         "4.25": "20250127_174840",
    #         "2.125": "20250127_175358",
    #     },
    #     "TLDR": {
    #         # "8.5": "20250127_175501",
    #         "4.25": "20250127_120947",
    #         "2.125": "20250127_162442",
    #     },
    # }
    
    # for dataset in dataset_batch_id_table:
    #     for reward_threshold in dataset_batch_id_table[dataset]:
    #         args.dataset = dataset
    #         args.batch_id = dataset_batch_id_table[dataset][reward_threshold]
    #         args.reward_threshold = reward_threshold
    #         main(args)
