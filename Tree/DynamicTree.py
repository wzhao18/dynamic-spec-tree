import torch
from torch.nn.functional import softmax
from .Tree import Tree
import time
from Engine.Engine import GraphInferenceEngine, GraphInferenceEngineTG
from utils import get_sampling_logits, ChildrenAccept, get_residual, sampling_without_replacement, _make_causal_mask, get_residual
from transformers import AutoTokenizer

class DynamicTree:
    def __init__(self, 
                 draft_model_engine :GraphInferenceEngine,
                 target_model_engine :GraphInferenceEngineTG,
                 prefix :torch.LongTensor,
                 temperature :float = 0.6,
                 top_p: float = 0.9,
                 max_length = 256,
                 device :str = 'cpu',
                 vocab_size = 32000,
                 tree_size = 64):
        self.draft_model_engine = draft_model_engine
        self.target_model_engine = target_model_engine
        self.prefix = prefix
        self.temperature = temperature
        self.draft_temperature = 0.6
        self.top_p = top_p
        self.max_length = max_length
        self.device = device
        self.vocab_size = vocab_size
        self.tree_size = tree_size
        self.dtype = torch.float16

        self.num_nodes = len(prefix)
        self.ground_truth_len = len(prefix)
        self.ground_truth = prefix

        # print(f"prefix: {prefix}")
        # print(f"self.num_nodes: {self.num_nodes}")

        # generated + speculated tokens
        self.tokens = torch.zeros(max_length, device=device).long()
        self.tokens[:len(prefix)] = prefix.to(self.device)
        
        position_ids = torch.arange(len(prefix)).to(self.device)
        self.storage_ids = torch.arange(self.max_length).to(self.device)
        
        attn_mask = _make_causal_mask((1, self.max_length), dtype=self.dtype, device=self.device)

        # print(f"input_ids.shape: {self.tokens[:len(prefix)].unsqueeze(0).shape}")
        # print(f"storage_ids.shape: {self.storage_ids[:len(prefix)].shape}")
        # print(f"position_ids.shape: {position_ids.unsqueeze(0).shape}")
        # print(f"attn_mask.shape: {attn_mask[:len(prefix)].shape}")

        draft_model_outputs = self.draft_model_engine.inference(
            input_ids=self.tokens[:len(prefix)].unsqueeze(0), 
            storage_ids=self.storage_ids[:len(prefix)], 
            position_ids=position_ids.unsqueeze(0),
            attn_mask=attn_mask[:len(prefix)][None, None, :, :]
        )
        
        self.draft_logits = torch.zeros((self.max_length, vocab_size), dtype=self.dtype).to(self.device)
        self.draft_logits[0] = draft_model_outputs[...,-1,:][0]

        self.draft_kv_len = len(prefix)
        self.target_kv_len = 0

        self.reset_tree()

        self.r = torch.rand(self.max_length, dtype=self.dtype).to(self.device)

        # for token sampling
        self.rand = torch.empty((self.tree_size, self.draft_logits.shape[1]), dtype=self.dtype).uniform_().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        prefix_string = self.decode_tokens(self.prefix)
        # print(f"prefix_string: {prefix_string}")

    def reset_tree(self):
        self.node_id_to_seq = {0: self.ground_truth}

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

    def decode_tokens(self, tokens):
        _str = self.tokenizer.decode(tokens, skip_special_tokens=True)
        return _str

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
        sampling_q = softmax(self.draft_logits[layer_node_indices] / self.draft_temperature, dim=-1)

        # print("=========================================")
        # print(f"grow step {grow_step} layer tokens:")

        # iterate over each node and sample its children
        for i in range(len(layer_node_indices)):

            node_idx = layer_node_indices[i]

            # print(f"node_idx: {node_idx}")

            subtree_size = self.subtree_sizes[node_idx]
            if grow_step == 0:
                num_descandents = subtree_size
            else:
                num_descandents = subtree_size - 1

            if num_descandents <= 0:
                layer_branch.append(0)
                self.children.append([])
                continue

            logit = sampling_q[i]

            # top_scores, top_indices = torch.topk(logit, k=5)

            # sequence = self.node_id_to_seq[node_idx]
            # print(f"Sequence: {self.decode_tokens(sequence)}")

            # print("Top 5 Confidence Scores:")
            # for score, idx in zip(top_scores, top_indices):
            #     print(f"Token: {self.decode_tokens(idx)} Score: {score.item():.4f}")

            # print(f"num_descandents: {num_descandents}")

            scores, sorted_indices = torch.topk(logit, k=num_descandents)
            token_ids = sorted_indices

            remain_quota = num_descandents
            curr_idx = 0
            num_children = 0
            child_tree_sizes = []

            while remain_quota > 0:
                score = scores[curr_idx]
                child_tree_size = max(min(torch.ceil(score * num_descandents).int().item(), remain_quota), 1)
                child_tree_sizes.append(child_tree_size)

                num_children += 1
                remain_quota -= child_tree_size

            # if num_descandents > 0:
            #     confidence_cutoff = 1 / num_descandents
            # else:
            #     confidence_cutoff = 100

            # print(f"confidence_cutoff: {confidence_cutoff}")

            # mask = logit >= confidence_cutoff
            # token_ids = torch.nonzero(mask, as_tuple=True)[0]

            # scores = logit[mask]
            # scores, sorted_indices = torch.sort(scores, descending=True)
            # token_ids = token_ids[sorted_indices]

            # num_children = token_ids.size(0)
            layer_branch.append(num_children)

            # child_tree_sizes = torch.floor(scores * num_descandents).int().tolist()
            next_layer_tree_sizes.extend(child_tree_sizes)

            if next_layer_node_indices:
                last_node_idx = next_layer_node_indices[-1]
            else:
                last_node_idx = self.node_indices[-1][-1]

            children_node_indices = list(range(last_node_idx + 1, last_node_idx + 1 + num_children))

            # for i in range(num_children):
            #     self.node_id_to_seq[children_node_indices[i]] = torch.cat([self.node_id_to_seq[node_idx], token_ids[i].unsqueeze(0)])

            next_layer_node_indices.extend(children_node_indices)
            self.children.append(children_node_indices)

            # for i in range(num_children):
                # print(f"\tCandidate Token: `{self.decode_tokens(token_ids[i])}`")

            self.tokens[self.num_nodes: self.num_nodes + num_children] = token_ids[:num_children]
            self.num_nodes += num_children

            # print(f"self.num_nodes: {self.num_nodes}")

            for child_node_idx in children_node_indices:

                if grow_step > 0:
                    # inherit parent mask
                    self.tree_mask[child_node_idx - 1] = self.tree_mask[node_idx - 1]

                # attend itself
                self.tree_mask[child_node_idx - 1][child_node_idx - 1] = 0.

                self.depths.append(grow_step + 1)

            # print("=========================================")
            # print()

        if not next_layer_node_indices:

            # print(f"self.node_indices: {self.node_indices}")
            # print(f"self.subtree_sizes: {self.subtree_sizes}")
            # print(f"self.children: {self.children}")
            # print(f"self.depths: {self.depths}")
            # print(f"self.layer_branches: {self.layer_branches}")

            return
        
        self.node_indices.append(next_layer_node_indices)
        self.subtree_sizes.extend(next_layer_tree_sizes)
        self.layer_branches.append(layer_branch)
        
        # print(f"self.node_indices: {self.node_indices}")
        # print(f"self.subtree_sizes: {self.subtree_sizes}")
        # print(f"self.children: {self.children}")
        # print(f"self.depths: {self.depths}")
        # print(f"self.layer_branches: {self.layer_branches}")

        next_layer_num_nodes = len(next_layer_node_indices)
        start_pos = self.num_nodes - next_layer_num_nodes
        end_pos = self.num_nodes
        start_node_idx = next_layer_node_indices[0]
        end_node_idx = next_layer_node_indices[-1] + 1

        position_ids = torch.zeros(next_layer_num_nodes).long().to(self.device) + self.ground_truth_len + grow_step
        
        # initially all masked out (-inf)
        attn_mask = torch.full(
            (next_layer_num_nodes, self.max_length),
            fill_value=torch.finfo(self.dtype).min,
            dtype=self.dtype,
            device=self.device
        )

        # attend to previous tokens
        attn_mask[:, :self.ground_truth_len] = 0

        # attention between tree tokens (note: skip root node)
        attn_mask[:, self.ground_truth_len:self.ground_truth_len + self.tree_size] = self.tree_mask[start_node_idx - 1:end_node_idx - 1, :]

        # print(f"self.num_nodes: {self.num_nodes}")
        # print(f"next_layer_num_nodes: {next_layer_num_nodes}")
        # print(f"start_pos: {start_pos}")
        # print(f"end_pos: {end_pos}")
        # print(f"start_node_idx: {start_node_idx}")
        # print(f"end_node_idx: {end_node_idx}")

        # print(f"self.tree_mask: {self.tree_mask}")

        # print(f"input_ids: {self.tokens[start_pos:end_pos].unsqueeze(0)}")
        # print(f"position_ids: {position_ids.unsqueeze(0)}")
        # print(f"attn_mask.shape: {attn_mask[None, None, :, :].shape}")
        # print(f"attn_mask: {attn_mask[:, :30][None, None, :, :]}")
        # print(f"storage_ids: {self.storage_ids[start_pos:end_pos]}")

        draft_model_outputs = self.draft_model_engine.graph_inference(
            input_ids = self.tokens[start_pos:end_pos].unsqueeze(0),
            position_ids = position_ids.unsqueeze(0),
            attn_mask = attn_mask[None, None, :, :],
            storage_ids=self.storage_ids[start_pos:end_pos]
        )

        self.draft_logits[start_node_idx:end_node_idx] = draft_model_outputs

        self.draft_kv_len = self.num_nodes
    
    @torch.inference_mode()
    def accept_step(self, parent_id):
        logits_id = parent_id - (self.ground_truth_len - 1)
        p = self.target_logits[logits_id]
        draft_logits = self.draft_logits[logits_id]
        
        children = self.children[logits_id]
        if len(children) == 0:
            return (-1, p)
        
        for idx, pos in enumerate(children):

            token = self.tokens[pos + (self.ground_truth_len - 1)]
            q = softmax(draft_logits / self.temperature, dim=-1)
            r = self.r[pos + (self.ground_truth_len - 1)]
            
            if p[token] > r * q[token]:
                # self.accept_idx_map[idx] += 1
                return (pos + (self.ground_truth_len - 1), None)
                # return (-1, p)
            else:
                p = get_residual(p, q)
                draft_logits[token] = torch.finfo(self.dtype).min
        # self.accept_idx_map[-1] += 1
        return (-1, p)

    @torch.inference_mode()
    def verify(self):

        if self.target_kv_len == 0:
            new_node_num = self.num_nodes - self.ground_truth_len
            start_pos = 0
            end_pos = self.num_nodes

            attn_mask = torch.full(
                (self.num_nodes, self.num_nodes),
                fill_value=torch.finfo(self.dtype).min,
                dtype=self.dtype,
                device=self.device
            )

            # prefix causal mask
            attn_mask[:self.ground_truth_len, :self.ground_truth_len] = _make_causal_mask((1, self.ground_truth_len), dtype=self.dtype, device=self.device)

            # attend to previous tokens
            attn_mask[self.ground_truth_len:self.num_nodes, :self.ground_truth_len] = 0

            # attention between tree tokens (note: skip root node)
            attn_mask[self.ground_truth_len:self.num_nodes, self.ground_truth_len:self.num_nodes] = self.tree_mask[:new_node_num, :new_node_num]

            position_ids = torch.arange(self.num_nodes).to(self.device)
            position_ids[len(self.prefix) : self.num_nodes] = (torch.tensor(self.depths[1:]).to(self.device) + len(self.prefix) - 1)

            target_model_outputs = self.target_model_engine.inference(
                                        input_ids=self.tokens[start_pos:end_pos].unsqueeze(0), 
                                        position_ids = position_ids.unsqueeze(0),
                                        attn_mask=attn_mask[None, None, :, :], 
                                        storage_ids=self.storage_ids[start_pos:end_pos]
                                    )
            self.target_logits :torch.FloatTensor= target_model_outputs[0][self.ground_truth_len - 1:]
            
        else:
            new_node_num = self.num_nodes - self.target_kv_len
            start_pos = self.target_kv_len
            end_pos = self.num_nodes

            attn_mask = torch.full(
                (new_node_num, self.num_nodes),
                fill_value=torch.finfo(self.dtype).min,
                dtype=self.dtype,
                device=self.device
            )

            # attend to previous tokens
            attn_mask[:, :self.target_kv_len + 1] = 0

            # attention between speculated tree tokens
            attn_mask[1:, self.target_kv_len + 1:self.num_nodes] = self.tree_mask[:new_node_num - 1, :new_node_num - 1]

            position_ids = torch.zeros(new_node_num, dtype=torch.long).to(self.device)

            position_ids[0] = torch.tensor([self.target_kv_len]).to(self.device)
            position_ids[1 : self.num_nodes] = (torch.tensor(self.depths[1:]).to(self.device) + self.target_kv_len - 1)

            target_model_outputs = self.target_model_engine.inference(
                                        input_ids = self.tokens[start_pos : end_pos].unsqueeze(0), 
                                        position_ids = position_ids.unsqueeze(0),
                                        attn_mask = attn_mask[None, None, :, :],
                                        storage_ids=self.storage_ids[start_pos : end_pos])
            self.target_logits :torch.FloatTensor = target_model_outputs[0]
        
        # print(f"len(self.target_logits): {len(self.target_logits)}")
        # print(f"self.ground_truth_len: {self.ground_truth_len}")

        assert len(self.target_logits) == (self.num_nodes - self.ground_truth_len + 1)

        self.target_logits = get_sampling_logits(logits=self.target_logits, top_p=self.top_p, T=self.temperature, replicate=False)
        
        self.target_logits = softmax(self.target_logits / self.temperature, dim=-1)
        
        # print("=================Target Logits===================")

        # # print top 5 tokens
        # for i in range(len(self.target_logits)):

        #     sequence = self.node_id_to_seq[i]
        #     print(f"Sequence: {self.decode_tokens(sequence)}")

        #     top_scores, top_indices = torch.topk(self.target_logits[i], k=5)
        #     for score, idx in zip(top_scores, top_indices):
        #         print(f"Token: {self.decode_tokens(idx)} Score: {score.item():.4f}")

        #     print()

        # print("=================Target Logits end===================")
        # print()

        accept_list = list(range(self.ground_truth_len))

        terminal = False

        while True:
            parent_id = accept_list[-1]
            pos, res = self.accept_step(parent_id)
            if pos != -1:
                # print(f"Accepted Token Index {pos}: `{self.decode_tokens(self.tokens[pos].unsqueeze(0))}`") 
                accept_list.append(pos)
                if self.tokens[pos] == 0 or self.tokens[pos] == 2:
                    terminal = True
                    break
            else:
                # print("Rejected Token")
                residual = res
                break

        accept_length = len(accept_list)

        # accept_list is a list of position indices
        self.tokens[:accept_length] = self.tokens[accept_list]

        if not terminal:
            if torch.isnan(residual).any():
                terminal = True
            else:
                self.tokens[accept_length] = residual.multinomial(num_samples=1, replacement=True)
                # print(f"Sampled bonus token: `{self.decode_tokens(self.tokens[accept_length].unsqueeze(0))}`")

        self.draft_model_engine.engine.kv_cache.gather_kv_incremental(accept_list[self.ground_truth_len:], self.ground_truth_len)
        self.target_model_engine.engine.kv_cache.gather_kv_incremental(accept_list[self.ground_truth_len:], self.ground_truth_len)

        if not terminal:
            self.prepare_for_next_iter(accept_list, self.tokens[:accept_length+1])
            return self.tokens[:accept_length+1], terminal
        else:
            return self.tokens[:accept_length], terminal
    
    def prepare_for_next_iter(self, accept_list: list[int], valid_tokens :torch.LongTensor):
        
        if len(accept_list) + 1 > self.max_length:
            return 

        self.ground_truth = valid_tokens
        self.ground_truth_len = len(valid_tokens)
        self.num_nodes = len(valid_tokens)

        attn_mask = torch.full(
            (1, self.max_length),
            fill_value=torch.finfo(self.dtype).min,
            dtype=self.dtype,
            device=self.device
        )
        attn_mask[0, :len(valid_tokens)] = 0

        accept_length = len(accept_list)

        draft_model_outputs = self.draft_model_engine.graph_inference(
                                    input_ids = self.tokens[accept_length].unsqueeze(0).unsqueeze(0), 
                                    storage_ids=self.storage_ids[accept_length].unsqueeze(0),
                                    position_ids=torch.tensor([accept_length], device=self.device).unsqueeze(0),
                                    attn_mask=attn_mask[None, None, :, :])

        self.draft_logits[0] = draft_model_outputs[0]
        self.draft_kv_len = len(valid_tokens)
        self.target_kv_len = accept_length

        # print(f"self.target_kv_len: {self.target_kv_len}")

        self.reset_tree()

        # print("=================Draft Bonus Logit===================")
        # print(f"Sequence: {self.decode_tokens(valid_tokens)}")

        # top_scores, top_indices = torch.topk(softmax(self.draft_logits[0] / self.temperature, dim=-1), k=5)
        # for score, idx in zip(top_scores, top_indices):
        #     print(f"Token: {self.decode_tokens(idx)} Score: {score.item():.4f}")

        # print("=================Draft Bonus Logit End===================")
        # print()