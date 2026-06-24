import torch
from policies.base.policy import Policy

class ProbMaxPolicy(Policy):
    def __init__(self, generative_model, args_cli):
        super().__init__(generative_model, args_cli)

    def sample(self, **kwargs) -> torch.Tensor:
        reference_policy = kwargs.get('reference_policy')
        root_belief_action_weights = reference_policy.belief_action_weights(0) 
        best_action_id = torch.argmax(root_belief_action_weights)
        return best_action_id
            