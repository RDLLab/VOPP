import torch

class Policy:
    def __init__(self, device, generative_model, args_cli):
        self._device = device                
        
    def __call__(self, **kwargs) -> torch.Tensor:        
        return self.sample(**kwargs)

    def sample(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError("'sample' not implemented")

    def prob(
        self,
        belief_node: torch.Tensor, 
        action: torch.Tensor, 
        normalized_visits: torch.Tensor = None
    ) -> torch.Tensor:
        raise NotImplementedError("'prob' not implemented")

    def reset(self):
        pass

    def update(self, **kwargs):
        pass

    def print(self):
        pass

    def init_belief_policy(self, current_nodes: torch.Tensor):
        pass

    @property
    def action_shape(self):
        raise NotImplementedError("Property 'action_shape' not implemented")