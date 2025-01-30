import json
import argparse
import numpy as np
from pathlib import Path


def analyze_batch(jsonl_path: str, batch_id: str):
    """
    Analyze statistics for a specific batch from a JSONL file.

    Args:
        jsonl_path: Path to the JSONL file
        batch_id: Batch ID to analyze

    Returns:
        tuple: (mean reward, stderr reward, total time, num_llm_call_lst, num_rm_call_lst)
    """
    rewards = []
    total_time = 0
    num_llm_call_lst = []
    num_rm_call_lst = []
    # Read and process the JSONL file
    with open(jsonl_path, "r") as f:
        for line in f:
            record = json.loads(line)
            if record["batch_id"] == batch_id:
                rewards.append(record["reward"])
                total_time = record["elapsed_time"]
                num_llm_call_lst.append(record["num_llm_call"])
                num_rm_call_lst.append(record["num_rm_call"])
            else:
                continue

    if not rewards:
        raise ValueError(f"No records found for batch_id: {batch_id}")

    # Calculate statistics
    mean_reward = np.mean(rewards)
    stderr_reward = np.std(rewards) / np.sqrt(len(rewards))

    return mean_reward, stderr_reward, total_time, num_llm_call_lst, num_rm_call_lst


def main():
    parser = argparse.ArgumentParser(
        description="Analyze batch statistics from JSONL file"
    )
    parser.add_argument(
        "jsonl_path",
        nargs="?",
        help="Path to the JSONL file",
        default='generated_responses_HH.jsonl'
        # default="generated_responses_TLDR.jsonl",
    )
    parser.add_argument(
        'batch_id', 
        nargs='?', 
        help='Batch ID to analyze', 
        # default='20250126_124451', # HH
        default='20250126_123605' # TLDR
    )

    args = parser.parse_args()

    try:
        mean_reward, stderr_reward, total_time, num_llm_call_lst, num_rm_call_lst = (
            analyze_batch(args.jsonl_path, args.batch_id)
        )
        # Convert seconds to HH:MM:SS format
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        print(f"File: {args.jsonl_path}")
        print(f"Batch ID: {args.batch_id}")
        print(f"Mean reward: {mean_reward:.4f}")
        print(f"Standard error reward: {stderr_reward:.4f}")

        print(
            f"Num LLM call: mean +/- stderr: {np.mean(num_llm_call_lst)} +/- {np.std(num_llm_call_lst) / np.sqrt(len(num_llm_call_lst)):.2f}"
        )
        print(
            f"Num RM call: mean +/- stderr: {np.mean(num_rm_call_lst)} +/- {np.std(num_rm_call_lst) / np.sqrt(len(num_rm_call_lst)):.2f}"
        )

        print(f"Total time: {time_str}")
    except FileNotFoundError:
        print(f"Error: File {args.jsonl_path} not found")
    except ValueError as e:
        print(f"Error: {e}")
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in file")


if __name__ == "__main__":
    main()
