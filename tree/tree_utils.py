import torch

def compute_obs_node_depths(tree):
    N = tree.observation_nodes.shape[0]
    parents = torch.full((N,), -1, dtype=torch.long, device=tree.device)

    # Step 1: get parent action node for each observation node
    parent_actions = tree.observation_nodes[:, 0].to(torch.long)

    # Step 2: get parent obs node for each action node (only valid where parent_actions != -1)
    valid = parent_actions >= 0
    parent_obs = tree.action_nodes[parent_actions[valid], 0].to(torch.long)

    # Step 3: assign grandparent obs as parent for each obs node
    parents[valid] = parent_obs

    # Now use your original depth propagation
    depths = torch.zeros(N, dtype=torch.long, device=tree.device)
    current = parents.clone()
    mask = current >= 0
    while mask.any():
        depths[mask] += 2  # each obs node is 2 layers down from its grandparent
        current[mask] = parents[current[mask]]
        mask = current >= 0

    return depths

def get_deepest_leaves(tree, depths=None):
    parents = tree.nodes[:, 0].to(torch.long)
    if depths is None:
        depths = compute_depths(parents)

    is_parent = torch.zeros_like(parents, dtype=torch.bool, device=tree.nodes.device)
    is_parent[parents[parents >= 0]] = True
    leaf_indices = (~is_parent).nonzero(as_tuple=True)[0]

    max_depth = depths.max()
    return leaf_indices[depths[leaf_indices] == max_depth]

def get_nodes_at_depth(depths, target_depth):
    return (depths == target_depth).nonzero(as_tuple=True)[0]