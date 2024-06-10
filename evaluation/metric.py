from tqdm import tqdm
import json
import argparse
from nltk import word_tokenize
import os
from simcse import SimCSE
import numpy as np
import nltk
nltk.download('punkt')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", default="./hh_rlhf_output/llama_7b_rs100.jsonl", type=str)
    parser.add_argument('--output_path', type=str, default='./eval_result/metric/llama_7b_rs100.json')
    parser.set_defaults(bottleneck=True)
    parser.set_defaults(augment=True)
    args = parser.parse_args()
    return args


def compute_rep_n(text, n):
    tokens = word_tokenize(text)
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    rep_n = 100 * (1.0 - len(set(ngrams)) / (len(ngrams) + 1))
    return rep_n


def compute_diversity(text):
    diversity = 1.0
    for n in range(2, 5):
        rep_n_val = compute_rep_n(text, n)
        diversity *= 1.0 - rep_n_val / 100
    return diversity


def clean(text, sep="###"):
    return text.split(sep)[0]


def average(entries):
    return sum(entries) / len(entries)


def compute_coherence(prompts, responses):
    model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
    similarities = np.array(model.similarity(prompts, responses))
    return similarities.trace() / len(similarities)


if __name__ == "__main__":
    args = get_args()

    
    entries = []
    with open(args.file_name, "r") as f:
        for line in tqdm(f):
            generation = json.loads(line)
            prompt = generation["prompt"]
            response = generation["response"]
            
            if len(response) == 0:
                response = " "
            rep_2 = compute_rep_n(response, 2)
            rep_3 = compute_rep_n(response, 3)
            rep_4 = compute_rep_n(response, 4)
            diversity = compute_diversity(response)
            entries.append(
                {
                    "prompt": prompt,
                    "response": response,
                    "original_response": generation["response"][len(prompt) :],
                    "rep_2": rep_2,
                    "rep_3": rep_3,
                    "rep_4": rep_4,
                    "diversity": diversity,
                    "response_length": len(response),
                }
            )

    evaluations = {
        "rep_2": average([entry["rep_2"] for entry in entries]),
        "rep_3": average([entry["rep_3"] for entry in entries]),
        "rep_4": average([entry["rep_4"] for entry in entries]),
        "diversity": average([entry["diversity"] for entry in entries]),
        "coherence": compute_coherence(
            [entry["prompt"] for entry in entries], [entry["response"] for entry in entries]
        ),
        "response_length": average([entry["response_length"] for entry in entries]),
        "entries": entries,
    }

    json.dump(evaluations, open(args.output_path, "w"), indent=2)