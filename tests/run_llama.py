import sys
sys.path.append("..")
from transformers import DataCollatorForLanguageModeling, AutoTokenizer
import torch
import numpy as np 
from datasets import load_from_disk
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from torch.nn.functional import softmax
from accelerate import Accelerator
import argparse
from data_converter import convert_wiki_dataset, convert_cnn_dataset, convert_c4_dataset_eval, convert_wikimqa_dataset
import argparse
from Tree.DynamicTree import DynamicTree
import time
from utils import get_sampling_logits, _make_causal_mask, cuda_graph_for_residual, cuda_graph_for_sampling_without_replacement, sampling_with_replacement_without_graphs, residual_without_graphs
from Engine.Engine import GraphInferenceEngine, GraphInferenceEngineTG
from Engine.offload_engine import OffloadEngine
import random

M = 256
model_name_or_path = "meta-llama/Llama-2-7b-hf"
# model_name_or_path = "JackFram/llama-68m"
T = 0.1
p = 1.0
seed = 17

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(seed)

target_model =  GraphInferenceEngineTG(max_length=M, model_name_or_path=model_name_or_path, dtype = torch.float16, device="cuda:0")

max_length = M
position_ids = torch.arange(max_length).to('cuda:0').unsqueeze(0)
storage_ids = torch.arange(max_length).to('cuda:0')
attn_mask = _make_causal_mask((max_length, max_length), target_model.dtype, target_model.device)
top_p = p

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

input_ids = tokenizer.encode("The future of AI is really not promising because", return_tensors='pt').to('cuda:0')
all_tokens = input_ids

inner_decoding_step = 0
start_length = 0
while inner_decoding_step < 32:
    if inner_decoding_step == 0:
        start_length = input_ids.shape[1]
        logits = target_model.inference(input_ids = input_ids, storage_ids=storage_ids[:start_length],
                                        position_ids = position_ids[..., :start_length], 
                                        attn_mask=attn_mask[:start_length, :start_length][None, None, :, :])[0][-1]
        
    else:
        logits = target_model.inference(input_ids = input_ids, storage_ids=storage_ids[start_length + inner_decoding_step-1 : start_length + inner_decoding_step],
                                        position_ids = position_ids[..., start_length + inner_decoding_step-1 : start_length + inner_decoding_step], 
                                        attn_mask=attn_mask[start_length + inner_decoding_step-1 : start_length + inner_decoding_step, :start_length + inner_decoding_step][None, None, :, :])[0][-1]

    logits = get_sampling_logits(logits=logits, top_p=top_p, T=T)
    p = softmax(logits / T, dim=-1)

    new_token = p.multinomial(num_samples=1).unsqueeze(0)
    input_ids = new_token
    inner_decoding_step += 1
    
    all_tokens = torch.cat((all_tokens, input_ids), dim=1)

    top_scores, top_indices = torch.topk(p, k=5)

    # Print the results
    print("Top 5 Confidence Scores:")
    for score, idx in zip(top_scores, top_indices):
        print(f"Token: {tokenizer.decode(idx, skip_special_tokens=True)} Score: {score.item():.4f}")

output_string = tokenizer.decode(all_tokens[0].tolist())

print(output_string)