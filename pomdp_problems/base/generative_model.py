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
            random.seed(args_cli.seed)
            np.random.seed(args_cli.seed)
            torch.manual_seed(args_cli.seed)
            torch.cuda.manual_seed(args_cli.seed)
            torch.cuda.manual_seed_all(args_cli.seed)

    def __call__(self, state: torch.Tensor, action: torch.Tensor, active_env_ids: torch.Tensor, **kwargs) -> dict:              
        output = self.sample(state, action, **kwargs)
        output['reward'].view(-1)[~active_env_ids] = 0.0 # Set dummy reward for terminal environments
        return output

    @property
    def num_actions(self) -> int:
        """ 
        Number of discrete actions in the action space.
        """
        raise NotImplementedError("num_actions not implemented")


    def reset(self):
        """ 
        Perform reset operations if needed.
        """
        pass


    def sample(self, state: torch.Tensor, action: torch.Tensor, **kwargs) -> dict:
        """ 
        Given a batch of states and actions, sample next states, observations,
        rewards, and termination flags.

        Args:
            state: torch.Tensor of shape (B, state_dim).
            action: torch.Tensor of shape (B,). Each entry is a discrete action ID.

        Returns:
            Dictionary containing:
                next_state: torch.Tensor of shape (B, state_dim).
                observation: torch.Tensor of shape (B, ...) or (B,).
                reward: torch.Tensor of shape (B,).
                terminal: torch.Tensor of shape (B,).
                info: Optional additional information.
        """
        raise NotImplementedError("'sample' not implemented")


    def likelihood(
        self, 
        observation: torch.Tensor,
        prev_state: torch.Tensor, 
        next_state: torch.Tensor, 
        action: torch.Tensor, 
        log_likelihood: bool = False,        
    ) -> torch.Tensor:
        """ 
        Compute the likelihood of an observation for a batch of candidate
        transitions.

        Args:
            observation: torch.Tensor containing the observation. This may be a
                single observation, e.g. shape (1,) for encoded observations, or
                another problem-specific observation shape.
            prev_state: torch.Tensor of shape (B, state_dim).
            next_state: torch.Tensor of shape (B, state_dim).
            action: torch.Tensor of shape (B,). Each entry is a discrete action ID.
            log_likelihood: If True, return log-likelihood values. Otherwise,
                return likelihood values.

        Returns:
            torch.Tensor of shape (B,) containing one likelihood or log-likelihood
            value per candidate transition.
        """
        raise NotImplementedError("'likelihood' not implemented")


    def sample_initial_belief(self, num_samples: int = 1) -> torch.Tensor:
        """ 
        Sample particles from the initial belief distribution.

        Args:
            num_samples: Number of particles to sample.

        Returns:
            torch.Tensor of shape (num_samples, state_dim).
        """
        raise NotImplementedError("'sample_initial_belief' not implemented")


    def heuristics(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor, 
        current_nodes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """ 
        Compute a heuristic value estimate for a batch of states and actions.

        Args:
            state: torch.Tensor of shape (B, state_dim).
            action: torch.Tensor of shape (B,). Each entry is a discrete action ID.
            current_nodes: Optional planner-specific node indices.

        Returns:
            torch.Tensor of shape (B,) containing one heuristic value per state-action pair.
        """
        raise NotImplementedError("'heuristics' not implemented")

    def is_goal(self, state: torch.Tensor) -> torch.Tensor:
        """ 
        Check whether a given state is a terminal state

        Args:
            state: torch.Tensor of shape (1, state_dimensions) - The state            

        Returns:
            torch.Tensor of shape (1,) - Boolean indicating whether the state is a goal state
        """
        return torch.zeros(1, dtype=torch.bool, device=state.device)

    def state_repr(self, state: torch.Tensor) -> str:
        """
        Returns a string representation of a state
        """
        return str(state.cpu())

    def action_repr(self, action: torch.Tensor) -> str:
        """
        Returns a string representation of an action
        """
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
