import torch
import torch.nn.functional as F
from policies.base.policy import Policy
from itertools import product
from utils.utils import filter_rows, matching_row_indices

class PreferencePolicy(Policy):
    def __init__(self, device, generative_model, args_cli):
        super().__init__(device, generative_model, args_cli)
        if not hasattr(args_cli, 'eta'):
            self.eta = 0.2
        else:
            self.eta = args_cli.eta

        action_ranges = generative_model.action_ranges
        actions = [list(p) for p in product(*action_ranges)]
        action_map = {i: torch.tensor(actions[i], dtype=torch.float32, device=self._device) for i in range(len(actions))}
        self.action_ids = torch.tensor(list(action_map.keys()), dtype=torch.long, device=self._device)
        self.action_values = torch.stack([action_map[k.item()] for k in self.action_ids])
        self.reset()

    @property
    def action_shape(self):
        return (len(self.action_ids),) 

    def belief_action_weights(self, belief):
        return self._belief_action_weights[0, 1:]

    def action_value(self, belief, action):
        return self.action_values[action]     

    def reset(self):
        root_distr = torch.ones((len(self.action_ids),), dtype=torch.float32, device=self._device) / len(self.action_ids)
        self._belief_action_weights = torch.cat((torch.tensor([0], dtype=torch.float32, device=self._device), root_distr), dim=0).view(1, -1)

    def _make_policy_distribution(self, belief_indices: torch.Tensor) -> torch.distributions.Categorical:
        belief_policies = self._belief_action_weights[belief_indices]
        weights = belief_policies[:, 1:]        
        weights_normalized = F.softmax(self.eta * weights, dim=1)            
        distr = torch.distributions.Categorical(logits=torch.log((weights_normalized) + 1e-12))        
        return distr 

    def get_initial_policy_for_beliefs(self) -> torch.Tensor:
        return None  

    def _compute_belief_value(self, belief_indices):
        weights = self._belief_action_weights[belief_indices, 1:]
        weights_max = torch.max(weights, dim=1)[0].view(-1, 1)            
        exp_weights = torch.exp(self.eta * (weights - weights_max))
        log_sum_weights = torch.log(torch.sum(exp_weights, dim=1) + 1e-12).view(-1, 1)
        belief_value = ((weights_max + (log_sum_weights / self.eta)).view(-1))
        return belief_value

    def sample(self, **kwargs) -> torch.Tensor:        
        belief_node = kwargs.get('belief_node')        

        # A: Add new policies for nodes in belief_node if they don't exist yet      
        belief_node_unique, inverse = torch.unique(belief_node, return_inverse=True)

        # A1: Filter out nodes for which we already have a policy        
        new_nodes = filter_rows(self._belief_action_weights, belief_node_unique.view(-1, 1), cols=[0])

        # A2: Add policies for new nodes
        new_belief_policies = self.get_initial_policy_for_beliefs()
        if new_belief_policies is None:
            # Revert to uniform policy
            new_belief_policies = torch.cat((
                new_nodes,#.to(dtype=torch.int64),
                torch.ones(
                    (new_nodes.shape[0], len(self.action_ids)), 
                    dtype=torch.float32, 
                    device=self._device
                ) / len(self.action_ids) # Add uniform policy for new nodes
            ), dim=1)

        self._belief_action_weights = torch.cat((
            self._belief_action_weights, 
            new_belief_policies,
        ), dim=0)

        # Get the policies at belief_node
        belief_indices = matching_row_indices(self._belief_action_weights, belief_node_unique.view(-1, 1), cols=torch.tensor([0], dtype=torch.int64, device=self._device))        
        belief_indices_expanded = belief_indices[inverse]
        distr = self._make_policy_distribution(belief_indices_expanded)

        # Sample from the policy distributions
        sampled_indices = distr.sample()        
        return {
            'action_id': self.action_ids[sampled_indices], # (N, )
            'action_value': self.action_values[sampled_indices] # (N, K)
        }

    def update(self, **kwargs):
        current_nodes = kwargs.get('current_nodes') # Unique nodes
        node_child_groups = kwargs.get('node_child_groups')
        actions = kwargs.get('actions')
        Q = kwargs.get('Q') # R(b, a) + \gamma * \sum_{o} P(o | b, a) * V(\tau(b, a, o))

        # Compute V_{k}(b) = Psi_max(b) + (1 / eta) * log (sum_a exp [\eta * (Psi(b, a) - Psi_max(b))])
        # 1.) Get the weights at b
        belief_indices = matching_row_indices(
            self._belief_action_weights, 
            current_nodes.view(-1, 1), 
            cols=torch.tensor([0], dtype=torch.int64, device=self._device)
        )

        vals_k = self._compute_belief_value(belief_indices) 
        self._belief_action_weights[belief_indices[node_child_groups], actions + 1] = (
            self._belief_action_weights[belief_indices[node_child_groups], actions + 1] - vals_k[node_child_groups] + Q
        )

        # Compute V_{k+1}(b)
        vals_k_plus_1 = self._compute_belief_value(belief_indices)       
        return vals_k_plus_1

    def to_str(self):
        s = ""
        weights = self._belief_action_weights[:1, 1:]
        #print(f"\nweights\n{weights}")
        k = min(10, weights[0].shape[0])
        values, indices = torch.topk(weights[0], k)
        s +=  "weights (sorted)\n" + str(values) + "\n"         
        probs = F.softmax(self.eta * weights, dim=1)
        s += "probs\n" + str(probs[0, indices]) + "\n"        
        #print(f"policy\n{probs}")
        s += f"prob max: {torch.max(probs)}"
        return s

    def print(self):        
        print(self.to_str())        
