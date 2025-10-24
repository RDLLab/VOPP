import time
import torch
from typing import Tuple

# Helper functions
def _num_obs_children( 
    observation_nodes: torch.Tensor, 
    unique_action_nodes: torch.Tensor,
) -> torch.Tensor:
    parent_ids = observation_nodes[:, 0].to(torch.long)  # (N_obs,)
    unique_action_nodes = unique_action_nodes.to(torch.long)

    # Sort parent_ids for searchsorted
    sorted_ids, _ = torch.sort(parent_ids)

    # Count how often each unique_action_node appears as a parent
    left = torch.searchsorted(sorted_ids, unique_action_nodes, side='left')
    right = torch.searchsorted(sorted_ids, unique_action_nodes, side='right')

    num_children = right - left  # shape: (U,)
    return num_children

def _select_existing_obs(    
    observation_nodes: torch.Tensor, 
    unique_actions: torch.Tensor, 
    not_widen_idx: torch.Tensor,
) -> torch.Tensor:
    action_subset = unique_actions[not_widen_idx].to(torch.long)  # (U-W,)
    parent_ids = observation_nodes[:, 0].to(torch.long)      # (N_obs,)

    # Sort parent_ids and keep sort index
    sorted_parents, sort_idx = torch.sort(parent_ids)
    
    # For each action_subset[i], find [left, right) bounds of its matches in sorted_parents
    left = torch.searchsorted(sorted_parents, action_subset, side='left')
    right = torch.searchsorted(sorted_parents, action_subset, side='right')
    counts = right - left

    # Sample uniformly within the bounds for each action
    rand_offsets = (torch.rand_like(counts.float()) * counts.float()).floor().to(torch.long)
    rand_offsets = rand_offsets.clamp(max=(counts - 1).clamp(min=0))
    sample_pos = left + rand_offsets

    # Map back to the original indices in observation_nodes
    selected_obs = sort_idx[sample_pos]  # ← this preserves correct mapping
    parent_check = parent_ids[selected_obs]

    return observation_nodes[selected_obs, 1].to(torch.long)

def _resample_helper(
    final_samples: torch.Tensor,
    node_states: torch.Tensor,
    observation_nodes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
    obs_ids_all = node_states[:, 0].to(torch.long)          # (N,) Observation IDs
    log_likelihoods_all = node_states[:, 1]                 # (N,) Log-likelihoods
    states_all = node_states[:, 2:]                         # (N, D) States

    # 1. Sort node_states by observation ID to group states
    sorted_obs_ids, sorted_idx = torch.sort(obs_ids_all)
    sorted_ll = log_likelihoods_all[sorted_idx]
    sorted_states = states_all[sorted_idx]

    # 2. Identify unique obs_ids and how many states correspond to each
    unique_obs_ids, counts = torch.unique_consecutive(sorted_obs_ids, return_counts=True)
    group_offsets = torch.cat([torch.tensor([0], device=counts.device), counts.cumsum(0)[:-1]])  # (G,)

    # 3. For each entry in final_samples, find its group index in unique_obs_ids    
    obs_ids = observation_nodes[final_samples, 0].to(torch.long)  # [B]
    sample_group_idx = torch.searchsorted(unique_obs_ids, obs_ids)

    # 4. Set up per-sample candidate indices for state selection
    max_group_size = counts.max()
    group_range = torch.arange(max_group_size, device=counts.device)  # [0, 1, ..., max-1]
    group_mask = group_range.unsqueeze(0) < counts[sample_group_idx].unsqueeze(1)  # (B, max)
    offsets = group_offsets[sample_group_idx].unsqueeze(1)                         # (B, 1)
    candidate_idxs = offsets + group_range                                         # (B, max)
    candidate_idxs = candidate_idxs.masked_fill(~group_mask, 0)                    # mask invalid rows

    # 5. Fetch log-likelihoods for each candidate
    candidate_ll = sorted_ll[candidate_idxs]                                       # (B, max)
    candidate_ll[~group_mask] = float('-inf')                                      # suppress masked
    return candidate_ll, candidate_idxs, sorted_states

def _resample_states(    
    final_samples: torch.Tensor,
    node_states: torch.Tensor,
    observation_nodes: torch.Tensor,
) -> torch.Tensor:    
    #candidate_ll, candidate_idxs, sorted_states = _resample_helper(final_samples, node_states, observation_nodes) 
    obs_ids_all = node_states[:, 0].to(torch.long)          # (N,) Observation IDs
    log_likelihoods_all = node_states[:, 1]                 # (N,) Log-likelihoods
    states_all = node_states[:, 2:]                         # (N, D) States

    # 1. Sort node_states by observation ID to group states
    sorted_obs_ids, sorted_idx = torch.sort(obs_ids_all)
    sorted_ll = log_likelihoods_all[sorted_idx]
    sorted_states = states_all[sorted_idx]

    # 2. Identify unique obs_ids and how many states correspond to each
    unique_obs_ids, counts = torch.unique_consecutive(sorted_obs_ids, return_counts=True)
    group_offsets = torch.cat([torch.tensor([0], device=counts.device), counts.cumsum(0)[:-1]])  # (G,)

    # 3. For each entry in final_samples, find its group index in unique_obs_ids    
    obs_ids = observation_nodes[final_samples, 0].to(torch.long)  # [B]
    sample_group_idx = torch.searchsorted(unique_obs_ids, obs_ids)

    # 4. Set up per-sample candidate indices for state selection
    max_group_size = counts.max()
    group_range = torch.arange(max_group_size, device=counts.device)  # [0, 1, ..., max-1]
    group_mask = group_range.unsqueeze(0) < counts[sample_group_idx].unsqueeze(1)  # (B, max)
    offsets = group_offsets[sample_group_idx].unsqueeze(1)                         # (B, 1)
    candidate_idxs = offsets + group_range                                         # (B, max)
    candidate_idxs = candidate_idxs.masked_fill(~group_mask, 0)                    # mask invalid rows

    # 5. Fetch log-likelihoods for each candidate
    candidate_ll = sorted_ll[candidate_idxs]                                       # (B, max)
    candidate_ll[~group_mask] = float('-inf') 

    # 6. Sample states based on softmax over log-likelihoods  
    probs = torch.softmax(candidate_ll, dim=1)                                     # (B, max)
    sampled_rel_idx = torch.multinomial(probs, num_samples=1).squeeze(1)          # (B,)    

    sampled_flat_idx = candidate_idxs.gather(1, sampled_rel_idx.unsqueeze(1)).squeeze(1)  # (B,)

    # 7. Return resampled states
    return sorted_states[sampled_flat_idx]

def _construct_observation_samples(    
    tree_observation_nodes: torch.Tensor,
    tree_action_nodes: torch.Tensor,
    current_nodes: torch.Tensor,
    observations: torch.Tensor,
    alpha_o: float,
    beta_o: float,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Setup
    unique_action_nodes, inverse = torch.unique(current_nodes, dim=0, return_inverse=True)
    U = unique_action_nodes.shape[0]
    obs_dim = observations.shape[1]

    # 1. Group sampled observations by action node
    sort_idx = torch.argsort(inverse)
    grouped_obs = observations[sort_idx]        
    counts = torch.bincount(inverse[sort_idx], minlength=U)
    offsets = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])  # shape (U+1,)

    # 2. Get number of existing observation children for unique action nodes
    num_children = _num_obs_children(tree_observation_nodes, unique_action_nodes)

    # 3. Compute thresholds and widening mask
    visit_counts = tree_action_nodes[unique_action_nodes, 3]  # column 3 stores N(a)
    thresholds = beta_o * visit_counts.float().pow(alpha_o)
    widen_mask = num_children < thresholds  # shape (U,)

    # 4. Sample from NEW observations
    #widen_idx = widen_mask.nonzero(as_tuple=True)[0]
    widen_idx = widen_mask.nonzero().squeeze(1)
    widen_counts = counts[widen_idx]
    widen_offsets = offsets[widen_idx]
    rand_idx = (torch.rand_like(widen_counts.float()) * widen_counts.float()).floor().to(torch.long)        
    new_obs = grouped_obs[widen_offsets + rand_idx]  # (W, obs_dim)

    # 5. Assign new observation IDs
    next_id = tree_observation_nodes[:, 1].max().item() + 1
    new_ids = torch.arange(
        next_id,
        next_id + new_obs.shape[0],
        device=device,
        dtype=torch.int64,
    )

    # 6. Register tree row indices in self.observation_nodes
    new_tree_rows = torch.arange(
        tree_observation_nodes.shape[0],
        tree_observation_nodes.shape[0] + new_obs.shape[0],
        device=device,
        dtype=torch.long
    ).unsqueeze(1).float()

    new_obs_rows = torch.cat([new_tree_rows, new_obs], dim=1)

    # 7. Sample from existing tree observations using fast lookup
    #not_widen_idx = (~widen_mask).nonzero(as_tuple=True)[0]  # (U-W,)
    not_widen_idx = (~widen_mask).nonzero().squeeze(1) # (U-W,)
    existing_obs = _select_existing_obs(tree_observation_nodes, unique_action_nodes, not_widen_idx)  # (U-W,)        

    # 8. Stitch final result
    final_samples = torch.empty((unique_action_nodes.shape[0],), device=device, dtype=torch.int64)
    final_samples[widen_idx] = new_ids
    final_samples[not_widen_idx] = existing_obs
    final_samples = final_samples[inverse]

    return new_obs_rows, final_samples

class ProgressiveWidener:
    def __init__(self, obs_dim, likelihood_fn, alpha_o=1, beta_o=1, device=None):
        self.device = device
        self.obs_dim = obs_dim        
        self.likelihood_fn = likelihood_fn
        self.beta_o = beta_o
        self.alpha_o = alpha_o
        self.reset()

    def reset(self):
        self.observation_nodes = torch.empty((0, 1+self.obs_dim[0]), dtype=torch.float32, device=self.device)
        self.node_states = None        

    def widen(
        self, 
        tree, 
        current_nodes: torch.Tensor, 
        action: torch.Tensor, 
        observations: torch.Tensor, 
        states: torch.Tensor
    ):  
        new_obs_nodes, final_samples = _construct_observation_samples(
            tree.observation_nodes,
            tree.action_nodes, 
            current_nodes, 
            observations,
            self.alpha_o,
            self.beta_o,
            tree.device,
        )

        self.observation_nodes = torch.cat([self.observation_nodes, new_obs_nodes], dim=0)     

        # Map final_samples (row indices) to actual observation IDs
        obs_ids = self.observation_nodes[final_samples, 0].view(-1, 1).to(states)  # [B, 1]
        obs_vals = self.observation_nodes[final_samples][:, 1:]

        # Compute likelihoods p(o | s, a) for weighted sampling of states that belong
        # to observation nodes
        likelihoods = self.likelihood_fn(obs_vals, states, action, log_likelihood=True).unsqueeze(1)
    
        # Concatenate with states to form full entries
        entries = torch.cat([obs_ids, likelihoods, states], dim=1)  # [B, 1 + D]

        # Append to self.node_states
        if self.node_states is None:
            self.node_states = entries
        else:
            self.node_states = torch.cat([self.node_states, entries], dim=0)

        resampled = _resample_states(final_samples, self.node_states, self.observation_nodes)
        return final_samples, resampled