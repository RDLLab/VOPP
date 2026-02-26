import torch
from backup.base.backup import BackupFn
from tree.tree_utils import compute_obs_node_depths, get_nodes_at_depth

class RefSolverIterativeBackupFn(BackupFn):
    def __init__(self, reference_policy):
        super().__init__()
        self._reference_policy = reference_policy       

    def backup(self, tree, gamma=0.95):        
        # Initialization
        V_obs = torch.zeros(tree.observation_nodes.shape[0], dtype=torch.float32, device=tree.device)

        C_obs = tree.observation_nodes[:, 2]
        N_obs = tree.observation_nodes[:, 3]
        C_act = tree.action_nodes[:, 2]
        N_act = tree.action_nodes[:, 3]

        # Compute leaf values (only observation nodes can be leaves)
        is_parent = torch.zeros(tree.observation_nodes.shape[0], dtype=torch.bool, device=tree.device)
        parent_act_ids = tree.action_nodes[:, 0].to(torch.long)
        is_parent[parent_act_ids] = True
        leaf_obs_nodes = (~is_parent).nonzero(as_tuple=True)[0]
        V_obs[leaf_obs_nodes] = C_obs[leaf_obs_nodes] / (N_obs[leaf_obs_nodes] + 1e-8)

        # Compute depths of observation nodes
        obs_depths = compute_obs_node_depths(tree)

        # Precompute action depths and number of obs children
        action_parent_obs = tree.action_nodes[:, 0].to(torch.long)
        action_depths = obs_depths[action_parent_obs]
        obs_parents = tree.observation_nodes[:, 0].to(torch.long)
        num_obs_per_action = torch.bincount(obs_parents[1:], minlength=tree.action_nodes.shape[0])
        has_no_obs_child = num_obs_per_action == 0

        current_depth = obs_depths.max()
        current_obs_nodes = get_nodes_at_depth(obs_depths, current_depth)     

        while True:            
            # Get parent actions of current obs nodes
            parent_actions_from_obs = tree.observation_nodes[current_obs_nodes, 0].to(torch.long)            

            # Add terminal actions at the same level (actions without observation children)
            at_this_level = (action_depths == current_depth - 2)
            terminal_actions = (at_this_level & has_no_obs_child).nonzero(as_tuple=True)[0]
            parent_actions = torch.cat((parent_actions_from_obs, terminal_actions))

            unique_parent_actions = torch.unique(parent_actions, sorted=True)
            obs_child_groups = torch.searchsorted(unique_parent_actions, parent_actions_from_obs)

            # Compute N(a) as sum_{o in children(a)} N(a, o)
            # Required to get unbiased p(o | b, a) estimates
            dynamic_N_act = torch.zeros(
                unique_parent_actions.shape[0], dtype=torch.float32, device=tree.device
            )
            dynamic_N_act.scatter_add_(0, obs_child_groups, N_obs[current_obs_nodes])          

            # Weighted backup from observations to actions
            weighted_values = torch.zeros_like(dynamic_N_act)
            obs_vals = (N_obs[current_obs_nodes] / (dynamic_N_act[obs_child_groups] + 1e-8)) * V_obs[current_obs_nodes]
            weighted_values.scatter_add_(0, obs_child_groups, obs_vals)

            # Compute Q(b, a)
            R_b_a = C_act[unique_parent_actions] / (N_act[unique_parent_actions] + 1e-10)
            Q = R_b_a + gamma * weighted_values

            # Get parent observation nodes of those action nodes
            parent_obs_of_actions = tree.action_nodes[unique_parent_actions, 0].to(torch.long)
            unique_parent_obs, act_child_groups = torch.unique(parent_obs_of_actions, return_inverse=True)

            # Update policy and get V(b)
            new_vals = self._reference_policy.update(
                current_nodes=unique_parent_obs,
                node_child_groups=act_child_groups,
                actions=tree.action_nodes[unique_parent_actions, 1].to(torch.int32),
                Q=Q,
            )

            # Update V_obs
            V_obs[unique_parent_obs] = new_vals
            current_depth -= 2
            current_obs_nodes = get_nodes_at_depth(obs_depths, current_depth)

            if current_obs_nodes.shape[0] == 1:
                break        

        return V_obs 
