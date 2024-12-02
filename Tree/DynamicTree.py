import torch
from torch.nn.functional import softmax
from .Tree import Tree
import time
from Engine.Engine import GraphInferenceEngine, GraphInferenceEngineTG
from utils import get_sampling_logits, ChildrenAccept, get_residual, sampling_without_replacement_dynamic, _make_causal_mask

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
        
        position_ids = torch.arange(len(prefix)).to(self.device)
        self.storage_ids = torch.arange(self.max_length).to(self.device)
        
        attn_mask = _make_causal_mask((1, len(prefix)), dtype=self.dtype, device=self.device)

        draft_model_outputs = self.draft_model_engine.inference(input_ids=self.tokens.unsqueeze(0), 
                            storage_ids=self.storage_ids[:len(prefix)], 
                            position_ids=position_ids.unsqueeze(0),
                            attn_mask=attn_mask[None, None, :, :])
        
        self.draft_logits = torch.zeros((self.max_length, vocab_size), dtype=self.dtype).to(self.device)
        self.draft_logits[0] = draft_model_outputs[...,-1,:][0]

        # token tree
        self.node_indices = [[0]]
        self.subtree_sizes = [self.tree_size]
        self.children = []
        self.depths = [0]
        self.layer_branches = []
        
        self.tree_mask = torch.full(
            (self.tree_size, self.tree_size),
            fill_value=torch.finfo(self.dtype).min,
            dtype=self.dtype,
            device=self.device
        )
        self.tree_mask[0][0] = 0.

    def construct_grow_map(self):

        grow_step = 0
        while True:
            # track old number of layers
            num_layers = len(self.node_indices)

            # grow tree by one layer
            self.grow_tree_layer(grow_step)

            if len(self.node_indices) == num_layers:
                # tree is not growing anymore
                break

            grow_step += 1

    # State:
    #   node_indices = [[0]]
    #   subtree_sizes = [16]
    #   children: []
    #   depths: [0]
    #   layer_branches: []

    # Step 0 Sampling:
    #   next_layer_node_indices = [1, 2, 3]
    #   next_layer_tree_sizes = [10, 3, 2]
    #   layer_branch = [3]

    # State:
    #   node_indices = [[0], [1, 2, 3]]
    #   subtree_sizes = [16, 10, 3, 2]
    #   children: [[1, 2, 3]]
    #   depths: [0, 1, 1, 1]
    #   layer_branches = [[3]]

    # Step 1 Sampling:
    #   next_layer_node_indices = [4, 5, 6, 7, 8, 9]
    #   next_layer_tree_sizes = [5, 3, 1, 1, 1, 1]
    #   layer_branch = [3, 2, 1]

    # State:
    #   node_indices = [[0], [1, 2, 3], [4, 5, 6, 7, 8, 9]]
    #   subtree_sizes = [16, 10, 3, 2, 5, 3, 1, 1, 1, 1]
    #   children: [[1, 2, 3], [4, 5, 6], [7, 8], [9]]
    #   depths: [0, 1, 1, 1, 2, 2, 2, 2, 2, 2]
    #   layer_branches = [[3], [3, 2, 1]]

    # Step 2 Sampling:
    #   next_layer_node_indices = [10, 11, 12, 13, 14]
    #   next_layer_tree_sizes = [2, 1, 1, 1, 1]
    #   layer_branch = [3, 2, 0, 0, 0, 0]

    # State:
    #   node_indices = [[0], [1, 2, 3], [4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14]]
    #   subtree_sizes = [16, 10, 3, 2, 5, 3, 1, 1, 1, 1, 2, 1, 1, 1, 1]
    #   children: [[1, 2, 3], [4, 5, 6], [7, 8], [9], [10, 11, 12], [13, 14], [], [], [], [], []]
    #   depths: [0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
    #   layer_branches = [[3], [3, 2, 1], [3, 2, 0, 0, 0, 0]]

    # Step 3 Sampling:
    #   next_layer_node_indices = [15]
    #   next_layer_tree_sizes = [1]
    #   layer_branch = [1, 0, 0, 0, 0, 0]

    # State:
    #   node_indices = [[0], [1, 2, 3], [4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15]]
    #   subtree_sizes = [16, 10, 3, 2, 5, 3, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1]
    #   children: [[1, 2, 3], [4, 5, 6], [7, 8], [9], [10, 11, 12], [13, 14], [], [], [], [], [], [15], [], [], [], []]
    #   depths: [0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4]
    #   layer_branches = [[3], [3, 2, 1], [3, 2, 0, 0, 0, 0], [1, 0, 0, 0, 0]]

    # Step 4 Sampling:
    #   next_layer_node_indices = []
    #   next_layer_tree_sizes = []
    #   layer_branch = []

    # State:
    #   node_indices = [[0], [1, 2, 3], [4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15]]
    #   subtree_sizes = [16, 10, 3, 2, 5, 3, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1]
    #   children: [[1, 2, 3], [4, 5, 6], [7, 8], [9], [10, 11, 12], [13, 14], [], [], [], [], [15], [], [], [], [], []]
    #   depths: [0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4]
    #   layer_branches = [[3], [3, 2, 1], [3, 2, 0, 0, 0, 0], [1, 0, 0, 0, 0]]
    
    @torch.inference_mode()
    def grow_tree_layer(self, grow_step):

        next_layer_node_indices = []
        next_layer_tree_sizes = []
        layer_branch = []

        # nodes to expand
        layer_node_indices = self.node_indices[grow_step]

        # take softmax of logits
        sampling_q = softmax(self.draft_logits[layer_node_indices] / self.temperature, dim=-1)

        # iterate over each node and sample its children
        for i in range(len(layer_node_indices)):

            node_idx = layer_node_indices[i]
            subtree_size = self.subtree_sizes[node_idx]
            num_descandents = subtree_size - 1
            logit = sampling_q[i]

            confidence_cutoff = 1 / num_descandents

            mask = logit >= confidence_cutoff
            token_ids = torch.nonzero(mask, as_tuple=True)[1]
            scores = logit[mask]
            scores, sorted_indices = torch.sort(scores, descending=True)
            token_ids = token_ids[sorted_indices]

            num_children = token_ids.size(0)
            layer_branch.append(num_children)

            child_tree_sizes = torch.floor(scores).int().tolist()
            next_layer_tree_sizes.extend(child_tree_sizes)

            last_node_idx = self.node_indices[-1][-1]
            children_node_indices = range(last_node_idx + 1, last_node_idx + 1 + num_children)

            next_layer_node_indices.extend(children_node_indices)
            self.children.append(children_node_indices)
            self.depths.append(grow_step + 1)

            self.tokens[self.num_nodes: self.num_nodes + num_children] = token_ids
            self.num_nodes += num_children

            for child_node_idx in children_node_indices:

                # inherit parent mask
                self.tree_mask[child_node_idx] = self.tree_mask[node_idx]

                # attend itself
                self.tree_mask[child_node_idx][child_node_idx] = 0.

        if not next_layer_node_indices:
            return
        
        self.node_indices.append(next_layer_node_indices)
        self.subtree_sizes.extend(next_layer_tree_sizes)
        self.layer_branches.append(layer_branch)
        
        next_layer_num_nodes = len(next_layer_node_indices)
        start_pos = self.num_nodes - next_layer_num_nodes
        end_pos = self.num_nodes
        start_node_idx = next_layer_node_indices[0]
        end_node_idx = next_layer_node_indices[-1] + 1

        position_ids = torch.zeros(next_layer_num_nodes).long().to(self.device) + len(self.prefix) + grow_step - 1
        
        draft_model_outputs = self.draft_model_engine.graph_inference(
            input_ids = self.tokens[start_pos:end_pos].unsqueeze(0),
            position_ids = position_ids.unsqueeze(0),
            attn_mask = self.tree_mask[start_node_idx:end_node_idx][None, None, :, :],
            storage_ids=self.storage_ids[start_pos:end_pos]
        )
        self.draft_logits[start_node_idx:end_node_idx] = draft_model_outputs[0]
    
    # @torch.inference_mode()
    # def verify(self, benchmark = False):
    #     new_node_num = (self.num_nodes - self.ground_truth_len + 1)
    #     if self.target_kv_len == 0:
    #         start_pos = 0
    #         end_pos = self.num_nodes
    #         attn_mask = self.attn_mask[start_pos: end_pos, :end_pos]
    #         attn_mask = attn_mask[None, None, :, :].type(self.target_model_engine.dtype)
    #         if benchmark:
    #             torch.cuda.synchronize()
    #             t1 = time.time()
    #         target_model_outputs = self.target_model_engine.inference(input_ids = self.tokens[start_pos : end_pos].unsqueeze(0), 
    #                                 position_ids = self.position_ids[start_pos : end_pos].unsqueeze(0), attn_mask = attn_mask, 
    #                                 storage_ids=self.storage_ids[start_pos : end_pos])
    #         if benchmark:
    #             torch.cuda.synchronize()
    #             t2 = time.time()
    #         self.target_logits :torch.FloatTensor= target_model_outputs[0][self.ground_truth_len - 1:]
            
    #     else:
    #         start_pos = self.target_kv_len
    #         end_pos = self.num_nodes
    #         attn_mask = self.attn_mask[start_pos: end_pos, :end_pos]
    #         attn_mask = attn_mask[None, None, :, :].type(self.target_model_engine.dtype)
    #         if benchmark:
    #             torch.cuda.synchronize()
    #             t1 = time.time()
    #         target_model_outputs = self.target_model_engine.inference(input_ids = self.tokens[start_pos : end_pos].unsqueeze(0), 
    #                                     position_ids =self.position_ids[start_pos : end_pos].unsqueeze(0), attn_mask = attn_mask,
    #                                     storage_ids=self.storage_ids[start_pos : end_pos])
    #         if benchmark:
    #             torch.cuda.synchronize()
    #             t2 = time.time()
    #         self.target_logits :torch.FloatTensor = target_model_outputs[0][-(new_node_num):]
        
    #     assert len(self.target_logits) == (self.num_nodes - self.ground_truth_len + 1)

    #     self.target_logits = get_sampling_logits(logits=self.target_logits, top_p=self.top_p, T=self.temperature, replicate=False)
        
    #     self.target_logits = softmax(self.target_logits / self.temperature, dim=-1)
        
    #     accept_list = self.seq_to_use[:self.ground_truth_len]
        
    #     terminal = False
    #     # Recursively check if any children will be accepted
    #     # pos is the accepted position, res is the residual probability if no position is accepted (used to randomly sample)
    #     while True:
    #         parent_id = accept_list[-1]
    #         pos, res = self.accept_step(parent_id=parent_id)
    #         if pos != -1:
    #             accept_list.append(pos)
    #             if self.tokens[pos] == 0 or self.tokens[pos] == 2:
    #                  terminal = True
    #                  break
    #         else:
    #             residual = res
    #             break
    #     if benchmark:
    #         torch.cuda.synchronize()
    #         t3 = time.time()
    #     accept_length = len(accept_list)
    #     if not terminal:
    #         if torch.isnan(residual).any():
    #              terminal = True
    #         else:
    #             self.tokens[accept_length] = residual.multinomial(num_samples=1, replacement=True)

    #     # accept_list is a list of position indices
    #     self.tokens[:accept_length] = self.tokens[accept_list]

    #     self.draft_model_engine.engine.kv_cache.gather_kv_incremental(accept_list[self.ground_truth_len:], self.ground_truth_len)
    #     self.target_model_engine.engine.kv_cache.gather_kv_incremental(accept_list[self.ground_truth_len:], self.ground_truth_len)

    #     if not terminal:
    #         if benchmark:
    #             torch.cuda.synchronize()
    #             t4 = time.time()
    #             self.prepare_for_next_iter(accept_list, self.tokens[:accept_length+1])
    #             return self.tokens[:accept_length+1], accept_length, accept_length, t2 - t1, t3-t2, t4 - t3, terminal
    #         self.prepare_for_next_iter(accept_list, self.tokens[:accept_length+1])
    #         return self.tokens[:accept_length+1], accept_length, accept_length, terminal
    #     else:
    #          if benchmark:
    #             torch.cuda.synchronize()
    #             t4 = time.time()
    #             return self.tokens[:accept_length], accept_length, accept_length, t2 - t1, t3-t2, t4 - t3, terminal
    #          return self.tokens[:accept_length], accept_length, accept_length, terminal