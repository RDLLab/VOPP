import torch
from tree.tree import Tree
import time
import math
from progressive_widener import ProgressiveWidener

# Will be removed later
def print_rows(tensor):
    for i in range(tensor.shape[0]):
        print(f"{i}: {tensor[i]}")

def _current_depth_old(n_iters, max_iter, max_depth):
    return min(math.floor(((max_depth - 1) / (max_iter-1)) * n_iters + 1), max_depth)

def _current_depth(current_depth, n_iters, alpha=0.5, beta=1.0, max_depth=1000):
    #new_depth = current_depth
    new_depth = int(beta * (n_iters ** alpha) + 1)
    '''if current_depth <  v:
        new_depth += 1'''

    return min(new_depth, max_depth)

class ParallelRefSolver:
    def __init__(
            self, 
            args_cli,
            pomdp_model, 
            reference_policy, # Reference policy
            policy, # Action selection policy
            backup_fn, # Backup function to use
        ):
        self.args_cli = args_cli
        self.pomdp_model = pomdp_model
        self._reference_policy = reference_policy
        self._policy = policy
        self._backup_function = backup_fn
        if self.args_cli.continuous_observations:
            self._progressive_widener = ProgressiveWidener(
                self.pomdp_model.obs_shape,
                self.pomdp_model._generative_model.likelihood,
                alpha_o=self.args_cli.alpha_o, 
                beta_o=self.args_cli.beta_o,
                device=self.pomdp_model._device,                
            )

    def reset(self):        
        self.pomdp_model.reset()

    def plan(self):        
        # Initialize search tree
        tree = Tree(device=self.pomdp_model._device)

        # Initialize reference policy 
        self._reference_policy.reset()
        if self.args_cli.continuous_observations:
            self._progressive_widener.reset()      

        # Sample episodes until we've sampled max_sampled_episodes episodes,
        # or the max planning time per step has been reached     
        num_sampled_episodes = 0 
        time_start = time.time()
        n_iters = 0
        max_iter = math.ceil(self.args_cli.max_sampled_episodes / self.args_cli.num_envs)
        depth = 1        
        for _ in range(max_iter):
            # Determine maximum search depth for this iteration
            depth = _current_depth(
                depth, 
                n_iters, 
                alpha=self.args_cli.alpha_t, 
                beta=self.args_cli.beta_t, 
                max_depth=self.args_cli.max_search_depth
            )

            # Sample states from the initial current belief
            states = self.pomdp_model.sample_belief(learned=True)

            # Perform forward search
            self._search(tree, states, depth=depth)

            # Backup
            q_values = self._backup_function(tree, gamma=self.args_cli.discount_factor)

            # Check of planning for the current step is over
            num_sampled_episodes += states.shape[0]                   
            elapsed = time.time() - time_start
            if elapsed >= self.args_cli.planning_time_per_step:
                break

            n_iters += 1

        # Select action to execute in the environment
        action = self._policy(
            tree=tree, 
            q_values=q_values, 
            reference_policy=self._reference_policy
        )

        return action, {
            'num_sims': 0, 
            'policy': self._reference_policy, 
            'planning_stats_str': f"\nSearch finished in: {elapsed:.2f} seconds with {n_iters} iterations. Maximum depth {depth}.\nNum sampled episodes: {num_sampled_episodes}\n"
        }

    def _search(self, tree, state, depth=1):
        # Start from the root        
        current_nodes = torch.zeros((state.shape[0],), dtype=torch.int32, device=tree.device)
        self._reference_policy.init_belief_policy(current_nodes)
        current_depth = 0
        while current_depth != depth and current_nodes.shape[0] > 0:
            # Update visit count of current nodes
            tree.update_current_node_visit_count(current_nodes)

            # Sample action from the reference policy            
            sampled_action = self._reference_policy(state=state, belief_node=current_nodes, tree=tree)            
            action_id = sampled_action['action_id']
            action = sampled_action['action_value']

            # Simulate sampled actions            
            outputs = self.pomdp_model.step(state, action, debug=False)
            state = outputs['next_state']
            observation = outputs['observation']
            reward = outputs['reward']
            terminal = outputs['terminal']

            # Insert actions into tree
            current_nodes = tree.insert_action(
                current_nodes, 
                action_id, 
                reward, 
                terminal,
            )            

            # Filter out states, actions and observations from terminal states
            state = state[~terminal.view(-1)] 
            observation = observation[~terminal.view(-1)] 
            action = action[~terminal.view(-1)]

            # Early exit when all states are terminal
            if state.shape[0] == 0:                
                break                                           

            if self.args_cli.continuous_observations:                
                observation, state = self._progressive_widener.widen(
                    tree, 
                    current_nodes, 
                    action, 
                    observation, state
                )

            # Insert observations into tree
            current_nodes = tree.insert_observation(
                current_nodes, 
                observation,
            )

            # Init policy for new nodes (if required)
            self._reference_policy.init_belief_policy(current_nodes)            
            current_depth += 1 

        # Compute & update value-estimate of leaf nodes        
        value_estimate = self.pomdp_model.heuristics(state, action, current_nodes=current_nodes)
        tree.update_leaf_values(value_estimate, current_nodes)