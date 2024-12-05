import sys
sys.path.append(".")
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

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='JackFram/llama-68m', help='model')
parser.add_argument('--target', type=str, default='meta-llama/Llama-2-7b-hf', help='target model')
parser.add_argument('--dataset', type=str, default="cnn", help='dataset path')
parser.add_argument('--tree_size', type=int, default=64)
parser.add_argument('--start', type=int, default=0, help='start')
parser.add_argument('--end', type=int, default=200, help='end')
parser.add_argument('--T', type=float, default=0.6, help='temperature')
parser.add_argument('--draft_T', type=float, default=0.6, help='draft temperature')
parser.add_argument('--P', type=float, default=0.9, help='top_p')
parser.add_argument('--M', type=int, default=256, help='max length')
parser.add_argument('--seed', type=int, default=17, help='random seed')
parser.add_argument('--Mode', type=str, default="greedy", help='tree mode')
parser.add_argument('--offloading', action='store_true')
args = parser.parse_args()

args.model = 'JackFram/llama-68m'
args.target = 'meta-llama/Llama-2-7b-hf'
args.T = 0.5
args.draft_T = 0.7
args.P = 1
args.M = 512
args.dataset = 'cnn'
args.start = 0
args.end = 10


print(args)
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(args.seed)

def simulation_fast(target_model : GraphInferenceEngineTG, draft_model: GraphInferenceEngine, dataloader: DataLoader, T=0.6, top_p=0.9, max_length=512):

    num_decoding_steps = 0
    num_large_model_steps = 0
    total_time = 0

    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            input_ids = batch['input_ids'][..., :128]
            labels = batch['labels'][..., :128]

            terminate = False
            if labels[0][-1] == -100:
                terminate = True
            
            spectree = DynamicTree(
                draft_model,
                target_model,
                prefix=input_ids.squeeze(0),
                temperature=T,
                top_p=top_p,
                max_length=max_length,
                device='cuda:0',
                tree_size=args.tree_size
            )
            torch.cuda.synchronize()
            t1 = time.time()
            while input_ids.shape[1] < 256 and terminate == False:
                
                spectree.construct_grow_map()
                valid_tokens, terminate = spectree.verify()

                num_decoding_steps += (valid_tokens.shape[0] - input_ids.shape[1])
                num_large_model_steps += 1
                input_ids = valid_tokens.unsqueeze(0)

                if (input_ids[0][-1] == 2) or (input_ids[0][-1] == 0):
                    terminate = True

            print(f"Sentence: {spectree.decode_tokens(input_ids[0])}")

            torch.cuda.synchronize()
            t2 = time.time()
            total_time += (t2 - t1)
            draft_model.clear_kv()
            target_model.clear_kv()
    print("total time :{:.5f}s, latency :{:.5f}s, decoding step: {}, large model step: {}, {}".format(total_time, total_time / num_decoding_steps, num_decoding_steps, num_large_model_steps, num_decoding_steps / num_large_model_steps))
    return num_decoding_steps / num_large_model_steps


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
eval_list = list(range(200, 2000))
import random
random.shuffle(eval_list)

if args.dataset == 'openwebtext':
    tokenized_dataset_eval = load_from_disk("../dataset/openwebtext_eval").select(eval_list[args.start :args.end])
elif args.dataset == 'wiki':
    tokenized_dataset_eval = convert_wiki_dataset(tokenizer=tokenizer).select(eval_list[args.start :args.end])
elif args.dataset == 'cnn':
    tokenized_dataset_eval = convert_cnn_dataset(tokenizer=tokenizer).select(eval_list[args.start :args.end])
elif args.dataset == 'wikimqa':
    tokenized_dataset_eval = convert_cnn_dataset(tokenizer=tokenizer).select(eval_list[args.start :args.end])
else:
    tokenized_dataset_eval = convert_c4_dataset_eval(tokenizer=tokenizer).select(eval_list[args.start :args.end])

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
dataloader = DataLoader(tokenized_dataset_eval, batch_size=1, collate_fn=data_collator, shuffle=False)

draft_model = GraphInferenceEngine(max_length=args.M, model_name_or_path = args.model, dtype = torch.float16, device="cuda:0")
target_model =  GraphInferenceEngineTG(max_length=args.M, model_name_or_path = args.target, dtype = torch.float16, device="cuda:0")

accelerator = Accelerator()
dataloader = accelerator.prepare(dataloader)

simulation_fast(
    target_model=target_model,
    draft_model=draft_model,
    dataloader=dataloader,
    T=args.T,
    top_p=args.P,
    max_length=args.M
)