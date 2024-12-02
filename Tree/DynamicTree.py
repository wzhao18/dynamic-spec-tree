import torch
from torch.nn.functional import softmax
from .Tree import Tree
import time
from Engine.Engine import GraphInferenceEngine, GraphInferenceEngineTG
from utils import get_sampling_logits, ChildrenAccept, get_residual, sampling_without_replacement, _make_causal_mask

class DynamicTree(Tree):
    def __init__(self, 
                 draft_model_engine :GraphInferenceEngine,
                 target_model_engine :GraphInferenceEngineTG,
                 prefix :torch.LongTensor,
                 temperature :float = 0.6,
                 top_p: float = 0.9,
                 max_length = 256,
                 device :str = 'cpu',
                 vocab_size = 32000,
                 tree_size = 32):
        self.draft_model_engine = draft_model_engine
        self.target_model_engine = target_model_engine
        self.prefix = prefix
        self.temperature = temperature
        self.top_p = top_p
        self.max_length = max_length
        self.device = device
        self.vocab_size = vocab_size
        self.tree_size = tree_size
        self.dtype = torch.float16

        self.num_nodes = len(prefix)

        # generated + speculated tokens
        self.tokens = torch.zeros(max_length, device=device).long()
        self.tokens[:len(prefix)] = prefix.to(self.device)
        
        # position_ids = torch.zeros(max_length).long().to(self.device)
        position_ids = torch.arange(len(prefix)).to(self.device)
        storage_ids = torch.arange(len(prefix)).to(self.device)
        
        # attn_mask = torch.full((len(prefix), len(prefix)), torch.finfo(self.dtype).min, dtype=self.dtype, device=self.device)
        # attn_mask.fill_(torch.finfo(self.dtype).min)
        attn_mask = _make_causal_mask((1, len(prefix)), dtype=self.dtype, device=self.device)


        draft_model_outputs = self.draft_model_engine.inference(input_ids=self.tokens.unsqueeze(0), 
                            storage_ids=storage_ids, 
                            position_ids=position_ids.unsqueeze(0),
                            attn_mask=attn_mask[None, None, :, :])
        self.draft_logits = torch.zeros((self.max_length, vocab_size), dtype=self.dtype).to(self.device)
        self.draft_logits[0] = draft_model_outputs[...,-1,:][0]


    # determine the number of branches next layer according to the logits
    # update attention mask and position_ids accordingly 
    # return terminal or not
    @torch.inference_mode()
    def collective_grow_dynamic(self):

        new_tokens_set, confidence_scores = sampling_without_replacement(self.draft_logits[idx_list], self.rand[idx_list], total_branch, self.temperature)
        self.tokens[self.num_nodes: self.num_nodes + total_branch] = new_tokens_set[self.sample_gather_indices[grow_step]]
        if benchmark:
                    torch.cuda.synchronize()
                    t2 = time.time()
                    x1 += (t2 - t1)
            
        self.num_nodes = self.num_nodes + total_branch
        

        
        start_pos = self.num_nodes - total_branch
        end_pos = self.num_nodes
        attn_mask = self.attn_mask[self.num_nodes - total_branch: self.num_nodes]   # attn_mask from the previous tokens to current all output (+ total_branch)
        attn_mask = attn_mask[None, None, :, :]
        
        draft_model_outputs = self.draft_model_engine.graph_inference(
            input_ids = self.tokens[self.draft_kv_len: self.num_nodes].unsqueeze(0),
            position_ids = self.position_ids[start_pos : end_pos].unsqueeze(0),
            attn_mask = attn_mask,
            storage_ids=self.storage_ids[self.draft_kv_len: self.num_nodes]
            
        )
        self.draft_kv_len = self.num_nodes
        self.draft_logits[start_pos - self.ground_truth_len + 1:end_pos - self.ground_truth_len + 1] = draft_model_outputs[0][-total_branch:]
        if benchmark:
                    torch.cuda.synchronize()
                    t3 = time.time()
                    x2 += (t3 - t2)
        if benchmark:
            return n_branch_list, x1, x2
        return n_branch_list

    def construct_grow_map(self):
        while True:
            terminal = self.collective_grow_dynamic()   
            if terminal:
                break
        return None
    
    @torch.inference_mode()
    def verify(self, benchmark = False):
        new_node_num = (self.num_nodes - self.ground_truth_len + 1)
        if self.target_kv_len == 0:
            start_pos = 0
            end_pos = self.num_nodes
            attn_mask = self.attn_mask[start_pos: end_pos, :end_pos]
            attn_mask = attn_mask[None, None, :, :].type(self.target_model_engine.dtype)
            if benchmark:
                torch.cuda.synchronize()
                t1 = time.time()
            target_model_outputs = self.target_model_engine.inference(input_ids = self.tokens[start_pos : end_pos].unsqueeze(0), 
                                    position_ids = self.position_ids[start_pos : end_pos].unsqueeze(0), attn_mask = attn_mask, 
                                    storage_ids=self.storage_ids[start_pos : end_pos])
            if benchmark:
                torch.cuda.synchronize()
                t2 = time.time()
            self.target_logits :torch.FloatTensor= target_model_outputs[0][self.ground_truth_len - 1:]
            
        else:
            start_pos = self.target_kv_len
            end_pos = self.num_nodes
            attn_mask = self.attn_mask[start_pos: end_pos, :end_pos]
            attn_mask = attn_mask[None, None, :, :].type(self.target_model_engine.dtype)
            if benchmark:
                torch.cuda.synchronize()
                t1 = time.time()
            target_model_outputs = self.target_model_engine.inference(input_ids = self.tokens[start_pos : end_pos].unsqueeze(0), 
                                        position_ids =self.position_ids[start_pos : end_pos].unsqueeze(0), attn_mask = attn_mask,
                                        storage_ids=self.storage_ids[start_pos : end_pos])
            if benchmark:
                torch.cuda.synchronize()
                t2 = time.time()
            self.target_logits :torch.FloatTensor = target_model_outputs[0][-(new_node_num):]
        
        assert len(self.target_logits) == (self.num_nodes - self.ground_truth_len + 1)

        self.target_logits = get_sampling_logits(logits=self.target_logits, top_p=self.top_p, T=self.temperature, replicate=False)
        
        self.target_logits = softmax(self.target_logits / self.temperature, dim=-1)
        
        accept_list = self.seq_to_use[:self.ground_truth_len]
        
        terminal = False
        # Recursively check if any children will be accepted
        # pos is the accepted position, res is the residual probability if no position is accepted (used to randomly sample)
        while True:
            parent_id = accept_list[-1]
            pos, res = self.accept_step(parent_id=parent_id)
            if pos != -1:
                accept_list.append(pos)
                if self.tokens[pos] == 0 or self.tokens[pos] == 2:
                     terminal = True
                     break
            else:
                residual = res
                break
        if benchmark:
            torch.cuda.synchronize()
            t3 = time.time()
        accept_length = len(accept_list)
        if not terminal:
            if torch.isnan(residual).any():
                 terminal = True
            else:
                self.tokens[accept_length] = residual.multinomial(num_samples=1, replacement=True)

        # accept_list is a list of position indices
        self.tokens[:accept_length] = self.tokens[accept_list]

        self.draft_model_engine.engine.kv_cache.gather_kv_incremental(accept_list[self.ground_truth_len:], self.ground_truth_len)
        self.target_model_engine.engine.kv_cache.gather_kv_incremental(accept_list[self.ground_truth_len:], self.ground_truth_len)

        if not terminal:
            if benchmark:
                torch.cuda.synchronize()
                t4 = time.time()
                self.prepare_for_next_iter(accept_list, self.tokens[:accept_length+1])
                return self.tokens[:accept_length+1], accept_length, accept_length, t2 - t1, t3-t2, t4 - t3, terminal
            self.prepare_for_next_iter(accept_list, self.tokens[:accept_length+1])
            return self.tokens[:accept_length+1], accept_length, accept_length, terminal
        else:
             if benchmark:
                torch.cuda.synchronize()
                t4 = time.time()
                return self.tokens[:accept_length], accept_length, accept_length, t2 - t1, t3-t2, t4 - t3, terminal
             return self.tokens[:accept_length], accept_length, accept_length, terminal

