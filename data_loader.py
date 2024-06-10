import json

import numpy as np
from datasets import load_dataset

def QA_loader(data_dir:str, split:str, batch_size:int, head:int=None, max_len:int=1024):

    # load data
    data = load_dataset(data_dir, split=split)

    len_prompt = [len(p) for p in data['prompt']]
    len_response = [len(r) for r in data['response']]
    assert len(len_prompt) == len(len_response), 'Prompts and responses not aligned!'

    # remove long prompts
    len_prompt_cleaned, prompt, response = [], [], []
    for i in range(len(len_prompt)):
        if len_prompt[i] <= max_len and len_response[i] <= max_len:
            len_prompt_cleaned.append(len_prompt[i])
            prompt.append(data['prompt'][i])
            response.append(data['response'][i])
        if head is not None and len(prompt) >= head:
            break
    
    prompt, response = np.array(prompt), np.array(response)

    # sort by prompt length
    prompt_idx = np.argsort(len_prompt_cleaned)

    # yield data
    idx = batch_size
    while idx < len(prompt_idx):
        sorted_idx = prompt_idx[idx - batch_size:idx]
        yield prompt[sorted_idx].tolist(), response[sorted_idx].tolist()
        idx += batch_size

    sorted_idx = prompt_idx[idx - batch_size:]
    yield prompt[sorted_idx].tolist(), response[sorted_idx].tolist()

def HH_loader(data_dir:str, split:str, batch_size:int=1, head:int=None, max_len:int=1024):

    # load data
    data = load_dataset(data_dir, split=split)

    len_prompt = [len(p) for p in data['prompt']]
    len_chosen = [len(c) for c in data['chosen']]
    len_rejected = [len(r) for r in data['rejected']]

    # remove long prompts
    prompt, chosen, rejected = [], [], []
    for i in range(len(len_prompt)):
        if len_prompt[i] <= max_len and len_chosen[i] <= max_len and len_rejected[i] <= max_len:
            prompt.append(data['prompt'][i])
            chosen.append(data['chosen'][i])
            rejected.append(data['rejected'][i])
        if head is not None and len(prompt) >= head:
            break
    
    prompt, chosen, rejected = np.array(prompt), np.array(chosen), np.array(rejected)

    # yield data
    idx = batch_size
    while idx < len(prompt):
        yield prompt[idx - batch_size:idx].tolist(), chosen[idx - batch_size:idx].tolist(), rejected[idx - batch_size:idx].tolist()
        idx += batch_size

    yield prompt[idx - batch_size:].tolist(), chosen[idx - batch_size:].tolist(), rejected[idx - batch_size:].tolist()

if __name__=='__main__':
    data = QA_loader('Dahoas/full-hh-rlhf', split='test', batch_size=2, head=5)
    for prompt, response in data:
        print(prompt)
