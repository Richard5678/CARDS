import os
import random
import numpy as np
# import pandas as pd
import json
import sys
import evaluate

def clean(text, sep="###"):
    result = text.split(sep)[0]
    # print(result)
    return result if len(result) > 0 else " "

if __name__ == "__main__":

    # path = "Results/v2_TLDR_diversity.json"
    # generations_v2 = json.load(open(path, "r"))
    
    path = "generated_responses_TLDR_1.04_20250128_235911_10.jsonl"  # Changed extension to .jsonl
    
    # Read JSONL file
    generations_v2 = []
    with open(path, 'r') as f:
        for line in f:
            generations_v2.append(json.loads(line.strip()))
    
    rouge = evaluate.load('rouge')
    
    length = 10
    
    generation_v2_all = []
    
    for k in range(0, length):
        sys.stderr.write(f"sample {k} running\n")
        generation_v2_score = []
        
        for i in range(k*10, k*10 + 10):
            ref = generations_v2[i]["result"]
            for j in range(i+1, k*10 + 10):
                generation = generations_v2[j]["result"]
                temp = rouge.compute(predictions=[ref], references=[generation], use_aggregator=False)['rougeL']
                generation_v2_score.append(temp)
        
        generation_v2_all.append(np.mean(generation_v2_score))
        
    print(f"v2_score: {np.mean(generation_v2_all), np.std(generation_v2_all) / np.sqrt(len(generation_v2_all))}")




