import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

class GenerativeModel:
    def __init__(self, **kwargs):
        args_cli = kwargs.get('args_cli')
        self.role = kwargs.get('role')        
        if self.role == 'exec':
            #os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            random.seed(args_cli.seed)
            np.random.seed(args_cli.seed)
            torch.manual_seed(args_cli.seed)
            torch.cuda.manual_seed(args_cli.seed)
            torch.cuda.manual_seed_all(args_cli.seed)
            #torch.backends.cudnn.benchmark = False
            #torch.backends.cudnn.deterministic = True

    def __call__(self, state: torch.Tensor, action: torch.Tensor, active_env_ids: torch.Tensor, **kwargs) -> dict:              
        output = self.sample(state, action, **kwargs)
        output['reward'].view(-1)[~active_env_ids] = 0.0 # Set dummy reward for inactive environments
        return output

    @property
    def state_shape(self):
        raise NotImplementedError('state_shape not implemented')

    @property
    def action_shape(self):
        raise NotImplementedError('action_shape not implemented')

    @property
    def obs_shape(self):
        raise NotImplementedError('obs_shape not implemented')

    @property
    def action_ranges(self):
        raise NotImplementedError("Property 'action_ranges' not implemented")

    def reset(self):
        """ 
        Perform reset operations if needed
        """
        pass           

    def sample(self, state: torch.Tensor, action: torch.Tensor, **kwargs) -> dict:
        """ 
        Given a state and an action, sample a next state, observation, reward, and termination.

        Args:
            state: torch.Tensor of shape (N, state_dimensions) - The state
            action: torch.Tensor of shape (N, action_dimensions) - The action

        Returns:
            Dictionary containing:
                next_state: torch.Tensor of shape (N, state_dimensions) - The sampled next state
                observation: torch.Tensor of shape (N,) - The sampled observations
                reward: torch.Tensor of shape (N,) - The sampled reward
                terminal: torch.Tensor of shape (N,) - The termination
                nsteps: int - Number of environments steps taken
                info: Any - Additional information
        """
        raise NotImplementedError("'sample' not implemented")

    def likelihood(
        self, 
        observation: torch.Tensor,
        prev_state: torch.Tensor, 
        next_state: torch.Tensor, 
        action: torch.Tensor, 
        log_likelihood: bool = False,
        is_encoded_observation: bool = True,
    ) -> torch.Tensor:
        """ 
        Given the next state and actions, compute the likelihood of the given observation

        Args:
            observation: torch.Tensor of shape (1, observation_dimension) - The observation
            state: torch.Tensor of shape (N, state_dimensions) - The next states
            action: torch.Tensor of shape (N, action_dimensions) - The actions

        Returns:
            torch.Tensor of shape (N, 1) - The computed likelihood
        """
        raise NotImplementedError("'likelihood' not implmented")

    def sample_initial_belief(self, num_samples=1) -> torch.Tensor:
        """ 
        Sample states from the initial belief

        Args:
            num_samples: int - Number of states to sample           

        Returns:
            torch.Tensor of shape (N, state_dimensions) - The sampled states
        """
        raise NotImplementedError("'sample_initial_belief' not implemented")

    def heuristics(self, state: torch.Tensor, action: torch.Tensor, current_nodes: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ 
        Given a state compute a heuristic estimate of the value function.

        Args:
            state: torch.Tensor of shape (N, state_dimensions) - The state            

        Returns:
            torch.Tensor of shape (N,) - The heuristic estimate
        """
        raise NotImplementedError("'heuristics' not implemented")

    def is_goal(self, state: torch.Tensor) -> torch.Tensor:
        """ 
        Check whether a given state is a terminal state

        Args:
            state: torch.Tensor of shape (1, state_dimensions) - The state            

        Returns:
            torch.Tensor of shape (1) - Boolean indicating whether the state is a goal state
        """
        return torch.zeros(1, dtype=torch.bool, device=state.device)

    def state_repr(self, state: torch.Tensor) -> str:
        return str(state.cpu())

    def action_repr(self, action: torch.Tensor) -> str:
        return str(action.cpu())

    def get_info(self) -> str:
        return ""

    def plot(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor, 
        observation: torch.Tensor,
        belief: torch.Tensor
    ):
        pass

    def regenerate_particles(
        self, 
        belief_particles: torch.Tensor, 
        action: torch.Tensor, 
        observation: torch.Tensor
    ):
        return None, None

    def postprocess_belief_particles(
        self, 
        prior_belief_particles: torch.Tensor,
        action: torch.Tensor,
        observation: torch.Tensor,
        belief_particles: torch.Tensor,
    ):
        """ 
        Can be implemented to postprocess belief particles (e.g., to increase particle diversity)
        """
        return belief_particles

    def get_shared_data():
        """ 
        Returns any data that is shared across environments
        """
        return None

    def get_info(self):
        return ""