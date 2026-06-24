import sys
import os
import time
import torch
import math
import numpy as np
from pomdp_problems.base.generative_model import GenerativeModel
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from typing import Optional

def decode_action_ids(action_ids: torch.Tensor, num_rocks: int) -> torch.Tensor:
    """
    Converts action IDs of shape (B,) into (B, 2) tensor of agent actions.

    Each agent has the following actions:
        - 0: EAST
        - 1: NORTH
        - 2: SOUTH
        - 3: WEST
        - 4: SAMPLE
        - 5+i: SENSE rock i, for i in [0, num_rocks)

    The action ID encodes both agents' actions in base-(5 + num_rocks).

    Args:
        action_ids (Tensor): (B,) long tensor of action IDs.
        num_rocks (int): Number of rocks.

    Returns:
        actions (Tensor): (B, 2), where actions[:, i] ∈ [0, 5 + num_rocks - 1]
    """
    B = action_ids.shape[0]
    base = 5 + num_rocks

    agent1_actions = action_ids % base
    agent0_actions = action_ids // base

    return torch.stack([agent0_actions, agent1_actions], dim=1)  # (B, 2)

def _flatten_state(agent_pos: torch.Tensor, rock_good: torch.Tensor, rock_checked: torch.Tensor) -> torch.Tensor:
    """
    agent_pos:     (B, 2, 2)
    rock_good:     (B, m)
    rock_checked:  (B, m)
    Returns:
        state: (B, 4 + 2m)
    """
    B = agent_pos.shape[0]
    flattened = torch.cat([
        agent_pos.view(B, 4),
        rock_good,
        rock_checked
    ], dim=1)
    return flattened


def generate_rocksample_map(n: int, m: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate rock and agent start positions for MultiAgent RockSample using PyTorch tensors.

    Args:
        n (int): Grid size (n x n).
        m (int): Number of rocks.
        seed (Optional[int]): Random seed.

    Returns:
        rock_positions (Tensor): (m, 2) tensor of rock (x, y) positions.
        start_poses (Tensor): (2, 2) tensor of agent start (x, y) positions.
    """
    # Define start poses
    start_poses = torch.tensor([
        [0, n // 2 + 1],
        [0, n // 2 - 1]
    ], dtype=torch.long, device=device)

    occupied = set((x.item(), y.item()) for x, y in start_poses)

    rock_positions = []
    while len(rock_positions) < m:
        x = torch.randint(low=0, high=n, size=(1,)).item()
        y = torch.randint(low=0, high=n, size=(1,)).item()
        if (x, y) not in occupied:
            rock_positions.append((x, y))
            occupied.add((x, y))

    rock_positions = torch.tensor(rock_positions, dtype=torch.long, device=device)  # (m, 2)
    return rock_positions, start_poses

def step(states: torch.Tensor, actions: torch.Tensor, rock_positions: torch.Tensor, map_size: int, simulate_obs: Optional[bool] = True):
    B, D = states.shape
    m = rock_positions.shape[1]
    device = states.device
    TERMINAL = -1
    half_eff = 20.0

    state = states.clone()
    rewards = torch.zeros(B, device=device)
    observations = torch.zeros((B, 2), dtype=torch.long, device=device)

    agent_pos = state[:, :4].view(B, 2, 2)
    rock_state = state[:, 4:4 + m]
    rock_checked = state[:, 4 + m:4 + 2 * m]
    a = actions

    x = agent_pos[:, :, 0]
    y = agent_pos[:, :, 1]
    active = (x != TERMINAL)

    ### --- Move --- ###
    move_mask = (a < 4) & active
    dx = torch.tensor([1, 0, 0, -1], device=device)
    dy = torch.tensor([0, -1, 1, 0], device=device)
    dx_a = dx[a.clamp(max=3)]
    dy_a = dy[a.clamp(max=3)]
    new_x = x + dx_a
    new_y = y + dy_a

    is_east = (a == 0)
    east_exit = (x + 1 >= map_size) & is_east & move_mask
    x = torch.where(east_exit, TERMINAL, x)
    y = torch.where(east_exit, TERMINAL, y)
    rewards += east_exit.to(rewards.dtype).sum(dim=1) * 10

    in_bounds = (new_x >= 0) & (new_x < map_size) & (new_y >= 0) & (new_y < map_size)
    move_valid = move_mask & in_bounds & ~east_exit
    move_invalid = move_mask & ~in_bounds & ~east_exit
    x = torch.where(move_valid, new_x, x)
    y = torch.where(move_valid, new_y, y)
    rewards += move_invalid.to(rewards.dtype).sum(dim=1) * -100

    ### --- Sample --- ###
    for i in range(2):
        sample_mask = (a[:, i] == 4) & active[:, i]
        x_s = x[sample_mask, i]
        y_s = y[sample_mask, i]

        if x_s.numel() > 0:
            b_idx = sample_mask.nonzero(as_tuple=True)[0]
            rp = rock_positions[b_idx]
            matches = (x_s[:, None] == rp[:, :, 0]) & (y_s[:, None] == rp[:, :, 1])
            has_rock = matches.any(dim=1)
            rock_idx = matches.float().argmax(dim=1)
            rock_val = rock_state[b_idx, rock_idx]

            r = torch.where(
                has_rock,
                torch.where(rock_val.bool(), 10.0, -10.0),
                torch.full_like(rock_val.float(), -100.0)
            )
            rewards[b_idx] += r

            good_sampled = has_rock & (rock_val == 1)
            rock_state[b_idx[good_sampled], rock_idx[good_sampled]] = 0

    ### --- Sense --- ###
    sense_mask = (a > 4) & active
    if sense_mask.any():
        rock_idx = (a - 5).clamp_min(0)
        for i in range(2):
            b_mask = sense_mask[:, i]
            b_idx = rock_idx[:, i]
            batch_idx = b_mask.nonzero(as_tuple=True)[0]
            checked_idx = b_idx[b_mask]
            rock_checked[batch_idx, checked_idx] += 1

        if simulate_obs:
            rx = torch.gather(rock_positions[:, :, 0], 1, rock_idx)
            ry = torch.gather(rock_positions[:, :, 1], 1, rock_idx)
            dist = ((x - rx) ** 2 + (y - ry) ** 2).sqrt()
            eff = (1 + torch.pow(2, -dist / half_eff)) * 0.5
            rand = torch.rand_like(eff)
            truth = torch.gather(rock_state, 1, rock_idx)
            sensed = torch.where(rand < eff, truth, 1 - truth) + 1
            observations = torch.where(sense_mask, sensed, observations)

    agent_pos = torch.stack([x, y], dim=-1)
    dones = ((agent_pos[:, 0, 0] == TERMINAL) & (agent_pos[:, 1, 0] == TERMINAL))
    next_state = _flatten_state(agent_pos, rock_state, rock_checked)
    return next_state, rewards, dones, observations

def _postprocess_belief_particles(
    prior_belief_particles,
    action,
    observation,
    belief_particles,
    num_rocks,
    rock_positions
):
    B = belief_particles.shape[0]
    device = prior_belief_particles.device

    # Assuming the decode_action_ids function is correctly defined elsewhere and works
    a = decode_action_ids(action.view(-1), num_rocks)

    # Agent positions from the belief particles
    agent_pos = belief_particles[:, :4].view(B, 2, 2)
    
    # Create masks for sensed and sampled rocks
    actioned_rocks_mask = torch.zeros(num_rocks, dtype=torch.bool, device=device)

    # 1. Mask for scanned rocks
    sense_mask = (a > 4)
    scanned_rock_indices = (a - 5).clamp_min(0)
    scanned_rock_indices_flat = scanned_rock_indices[sense_mask]
    if scanned_rock_indices_flat.numel() > 0:
        actioned_rocks_mask[scanned_rock_indices_flat.unique().long()] = True

    # 2. Mask for sampled rocks
    sample_mask = (a == 4)
    if sample_mask.any():
        # Get the particle and agent indices for all sample actions
        sampled_indices = sample_mask.nonzero(as_tuple=True)
        sampled_particle_indices, sampled_agent_indices = sampled_indices

        # Get the positions of all agents who sampled
        sampled_agent_pos = agent_pos[sampled_particle_indices, sampled_agent_indices]

        # Corrected logic here to handle rock_positions shape (B, num_rocks, 2)
        # Get the rock positions corresponding to the sampled particles
        rock_pos_for_sampled_particles = rock_positions[sampled_particle_indices]
        
        # Check for matches between agent positions and rock positions
        matches = (sampled_agent_pos[:, None, 0] == rock_pos_for_sampled_particles[:, :, 0]) & \
                  (sampled_agent_pos[:, None, 1] == rock_pos_for_sampled_particles[:, :, 1])

        if matches.any():
            # Get the index of the sampled rock for each relevant particle
            sampled_rock_indices = matches.float().argmax(dim=1)
            actioned_rocks_mask[sampled_rock_indices.unique().long()] = True

    unscanned_rocks = ~actioned_rocks_mask

    # 1. Compute rock goodness probability for unscanned rocks
    prior_rock_states = prior_belief_particles[:, 4:4 + num_rocks]
    unscanned_rock_probs = prior_rock_states[:, unscanned_rocks].float().mean(dim=0)

    # 2. Resample the state of the unscanned rocks for all particles
    num_unscanned = unscanned_rock_probs.shape[0]
    random_samples = torch.rand(B, num_unscanned, device=device)
    new_unscanned_states = (random_samples < unscanned_rock_probs).to(prior_rock_states.dtype)
    
    # Create the tensor for the new rock states
    new_all_rock_states = belief_particles[:, 4:4 + num_rocks].clone()

    # Replace the states of only the unscanned rocks
    new_all_rock_states[:, unscanned_rocks] = new_unscanned_states
    
    # Update the belief particles
    belief_particles[:, 4:4 + num_rocks] = new_all_rock_states

    return belief_particles

def _apply_sense_far_penalty(value, action, agent_pos, rock_pos, alive_mask, num_rocks,
                             c=0.15, hinge=2):
    """
    Subtracts a penalty if an agent is scanning a rock that is far away.

    Args:
        value:      (B,) current heuristic values
        action:     (B,) encoded action IDs OR (B,2) decoded actions
        agent_pos:  (B,2,2) float, agent positions
        rock_pos:   (m,2) float, rock positions
        alive_mask: (B,2) bool, True if agent is alive
        num_rocks:  int
        c:          penalty scale
        hinge:      free steps before penalty starts (Manhattan distance)

    Returns:
        (B,) updated heuristic values with penalties applied
    """
    B = agent_pos.shape[0]
    m = num_rocks

    if m == 0:
        return value  # no rocks to sense

    actions = decode_action_ids(action.squeeze(-1), num_rocks)  
    is_sense = actions >= 5                               # (B,2)
    rock_idx = (actions - 5).clamp(min=0, max=m-1).long()        # (B,2)

    # Direct advanced indexing instead of gather
    target_pos = rock_pos[rock_idx]                      # (B,2,2)

    # Manhattan distance to target rock
    dist = (agent_pos - target_pos).abs().sum(dim=-1)    # (B,2)

    # Penalty only for distance beyond hinge
    over = (dist - float(hinge)).clamp_min(0.0)          # (B,2)
    pen = c * over                                       # (B,2)

    # Apply mask: only alive agents sensing
    pen = torch.where(is_sense & alive_mask, pen, torch.zeros_like(pen))

    # Subtract penalty from value
    return value - pen.sum(dim=1)


def _greedy_heuristic_simple(state, action, rock_positions, map_size, gamma):
    B = state.shape[0]
    m = rock_positions.shape[0]
    device = state.device
    _gamma = gamma

    rock_pos = rock_positions.to(device)  # (m, 2)

    # Unpack agent positions and rock states
    agent_pos = state[:, :4].view(B, 2, 2)    # (B, 2, 2)
    rock_state = state[:, 4:4 + m]            # (B, m), 0=bad, 1=good

    value = torch.zeros(B, device=device)

    # Distances: (B, 2, m)
    #agent_rock_dist = (agent_pos[:, :, None, :] - rock_pos[None, None, :, :]).abs().sum(dim=-1)
    agent_rock_dist = (agent_pos[:, :, None, :].float() - rock_pos[None, None, :, :]).abs().sum(dim=-1)

    # Mask out bad rocks
    is_good = rock_state.bool()               # (B, m)
    mask_bad_rocks = ~is_good[:, None, :]     # (B, 1, m)
    agent_rock_dist[mask_bad_rocks.expand(-1, 2, -1)] = float('inf')

    # Alive mask
    alive_mask = (agent_pos[:, :, 0] != -1)   # (B, 2)
    num_alive = alive_mask.sum(dim=1).clamp(min=1e-3)

    # Closest rock distance per agent
    min_dist_per_agent, min_idx = agent_rock_dist.min(dim=2)  # (B, 2)

    # Immediate rock reward
    rr = 12.0
    discounted = rr * (_gamma ** (min_dist_per_agent))  # (B, 2)
    sample_reward = max(rr * _gamma, 10.0)
    discounted = torch.where(
        min_dist_per_agent == 0,
        torch.full_like(discounted, sample_reward),
        discounted
    )
    discounted = torch.where(alive_mask, discounted, torch.zeros_like(discounted)) 
    rock_reward = discounted.sum(dim=1)  # (B,) 

    # Detect double-count: both alive, both at distance 0, same rock index
    both_alive = alive_mask.all(dim=1)                          # (B,)
    both_on_rock = (min_dist_per_agent == 0).all(dim=1)         # (B,)
    same_rock = (min_idx[:, 0] == min_idx[:, 1])                # (B,)
    double_mask = both_alive & both_on_rock & same_rock         # (B,)
    rock_reward = torch.where(double_mask, torch.full_like(rock_reward, rr), rock_reward)

    value += rock_reward# / num_alive
    if action is not None:
        value = _apply_sense_far_penalty(value, action, agent_pos.float(), rock_pos.float(),
                                     alive_mask, m, c=1.0, hinge=2)

    # --- Exit reward only for batch rows with no good rocks ---
    has_any_good = is_good.any(dim=1)                 # (B,)
    no_good_mask = (~has_any_good).unsqueeze(1)       # (B, 1)

    dx_exit = (map_size - 1 - agent_pos[:, :, 0]).clamp(min=0)  # (B, 2)
    goal_rewards = 10.0 * (_gamma ** (dx_exit))                    # (B, 2)
    goal_rewards = torch.where(alive_mask, goal_rewards, torch.zeros_like(goal_rewards))
    goal_rewards = goal_rewards * no_good_mask                  # zero where there IS a good rock

    value += goal_rewards.sum(dim=1)# / num_alive
    return value

def encode_action_ids(actions: torch.Tensor, num_rocks: int) -> torch.Tensor:
    """
    Converts a (B, 2) tensor of agent actions into a single action ID of shape (B,).

    Each agent has actions in [0, 5 + num_rocks - 1], and the combined action ID
    encodes agent0 and agent1's actions in base-(5 + num_rocks):

        action_id = agent0_action * base + agent1_action

    Args:
        actions (Tensor): (B, 2), where actions[:, i] ∈ [0, 5 + num_rocks - 1]
        num_rocks (int): Number of rocks.

    Returns:
        action_ids (Tensor): (B,) long tensor of action IDs.
    """
    base = 5 + num_rocks
    agent0_action = actions[:, 0]
    agent1_action = actions[:, 1]
    return agent0_action * base + agent1_action

def encode_observation(obs: torch.Tensor) -> torch.Tensor:
    """
    Encode 2D observations (B, 2) to integer observations (B,)
    Each entry is assumed to be in {0=NONE, 1=BAD, 2=GOOD}

    Encoding: obs[:, 0] * 3 + obs[:, 1] ∈ {0..8}
    """
    return obs[:, 0] * 3 + obs[:, 1]

def decode_observation(obs_id: torch.Tensor) -> torch.Tensor:
    """
    Decode integer observations (B,) to 2D observations (B, 2)
    Each value in obs_id ∈ {0..8}, and is decoded to (a, b) where:
        a = obs_id // 3
        b = obs_id % 3
    """
    return torch.stack([obs_id // 3, obs_id % 3], dim=-1)

def sample_initial_states(num_samples, rock_positions, start_poses):
    """
    Samples full initial states for MultiAgent RockSample.

    Each sample includes:
        - Agent positions: (x, y) for each agent
        - Rock goodness: 0 = bad, 1 = good
        - Rock checked flags: 0 = unchecked

    Returns:
        state: (num_samples, 4 + 2 * m)
    """
    m = rock_positions.shape[0]
    device = rock_positions.device

    # Agent starting positions
    agent_pos = start_poses.unsqueeze(0).expand(num_samples, -1, -1)  # (num_samples, 2, 2)

    # Rock state: 0 = bad, 1 = good
    rock_good = torch.randint(0, 2, size=(num_samples, m), dtype=torch.long, device=device)

    # Rock checked flags: all zeros initially
    rock_checked = torch.zeros((num_samples, m), dtype=torch.long, device=device)

    return _flatten_state(agent_pos, rock_good, rock_checked)


class MaRocksample(GenerativeModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gamma = kwargs.get('args_cli').discount_factor        
        self._device = kwargs.get('args_cli').device        
        self.headless = kwargs.get('args_cli').headless        

        self._num_agents = 2        
        self._num_rocks = kwargs.get('args_cli').num_rocks
        self._num_actions = int(pow(self._num_rocks + 5, self._num_agents))
        self._map_size = kwargs.get('args_cli').map_size
        self.fig, self.ax = None, None
        if self.role == 'planning':
            self.exec_env = kwargs.get('exec_env', None)    
        
    @property
    def num_actions(self) -> int:
        return self._num_actions

    def get_shared_data(self):
        return {'rock_positions': self.rock_positions, 'start_poses': self.start_poses}        

    def reset(self):        
        if self.role == 'exec':
            self.rock_positions, self.start_poses = generate_rocksample_map(self._map_size, self._num_rocks, self._device) 
        else:
            data = self.exec_env.get_shared_data()
            self.rock_positions = data['rock_positions'].to(device=self._device)
            self.start_poses = data['start_poses'].to(device=self._device)

    def sample_initial_belief(self, num_samples: int = 1) -> torch.Tensor:
        """
        Samples full initial states for MultiAgent RockSample.

        Each sample includes:
            - Agent positions: (x, y) for each agent
            - Rock goodness: 0 = bad, 1 = good
            - Rock checked flags: 0 = unchecked

        Returns:
            state: (num_samples, 4 + 2 * m)
        """
        initial_states = sample_initial_states(num_samples, self.rock_positions, self.start_poses)
        if self.role == "exec":
            num_good_rocks = initial_states[0, 4 : 4 + self.rock_positions.shape[0]].sum().item()
            self._num_good_rocks_initial = num_good_rocks 
            self._num_bad_rock_samplings = 0           
        return initial_states

    def sample(self, state: torch.Tensor, action: torch.Tensor, **kwargs) -> dict: 
        B = state.shape[0]
        actions = decode_action_ids(action.to(dtype=torch.int64).view(-1), num_rocks=self.rock_positions.shape[0])        
        next_state, reward, done, obs = step(state, actions, self.rock_positions.unsqueeze(0).expand(B, -1, -1), self._map_size) 

        if self.role == "exec":
            num_good_rocks = next_state[0, 4 : 4 + self.rock_positions.shape[0]].sum().item()
            self._num_good_rocks_current = num_good_rocks
            rew = reward[0].long().cpu().item() 
            if rew == -10:
                self._num_bad_rock_samplings += 1
            elif rew == -20:
                self._num_bad_rock_samplings += 2
        
        return {
            'next_state': next_state,
            'observation': encode_observation(obs),
            'reward': reward,
            'terminal': done,
        }

    def heuristics(self, state: torch.Tensor, action: torch.Tensor, current_nodes: Optional[torch.Tensor] = None) -> torch.Tensor:            
        B = state.shape[0]
        m = self.rock_positions.shape[0]
        rock_pos = self.rock_positions.unsqueeze(0).expand(B, -1, -1)  # (B, m, 2)        
        return _greedy_heuristic_simple(state, action, self.rock_positions, self._map_size, self.gamma)

    def likelihood(
        self, 
        observation: torch.Tensor,
        states: torch.Tensor, 
        next_state: torch.Tensor, 
        action: torch.Tensor,
        log_likelihood: bool = False,
        is_encoded_observation: bool = True,
    ) -> torch.Tensor:
        """
        Compute likelihood p(o | s', a) for each batch element.
        Assumes:
            - observation: (B,), values ∈ {0, 1, 2}
            - next_state: (B, 4 + m), where first 4 dims are agent positions
            - action: (B,), raw encoded action IDs
        Returns:
            - likelihood: (B,) p(o | s', a)
        """
        act = decode_action_ids(action, num_rocks=self._num_rocks).to(dtype=torch.int64)
        obs = decode_observation(observation)        
        B = action.shape[0]
        m = self.rock_positions.shape[0]
        half_eff = 20.0        

        agent_pos = next_state[:, :4].view(B, 2, 2).float()        # (B, 2, 2)
        rock_state = next_state[:, 4:]                            # (B, m), values ∈ {0,1}

        # Initialize likelihood to 1.0
        likelihood = torch.ones(B, dtype=torch.float32, device=next_state.device)

        for i in range(2):  # Loop over agents (small fixed number)
            a_i = act[:, i]              # (B,)
            obs_i = obs[:, i]            # (B,)
            pos = agent_pos[:, i, :]     # (B, 2)
            
            active_mask = (pos[:, 0] >= 0)#.view(-1)  # (B,)

            # Sense mask: where agent is active & sensing a rock
            a_i_mask = (a_i >= 5).view(-1)        # (B,)
            sense_mask = a_i_mask & active_mask   # (B,)

            rock_idx = (a_i - 5).clamp(min=0, max=m - 1).view(-1)

            # Compute distance from agent to sensed rock
            rock_pos = self.rock_positions[rock_idx]             # (B, 2)
            dists = torch.norm(pos - rock_pos, dim=1)            # (B,)

            # Compute efficiency
            eff = (1 + torch.pow(2, -dists / half_eff)) * 0.5    # (B,)

            # Get true rock status: 1=GOOD, 0=BAD
            true = rock_state[torch.arange(B), rock_idx]         # (B,)
            reported = (obs_i == 2).float()                      # (B,)  # 2=GOOD, 1=BAD

            # For sense actions, compute prob of reported outcome
            prob = torch.where(reported == true.float(), eff, 1 - eff)  # (B,)

            # Only apply where sense_mask is True
            likelihood = likelihood * torch.where(sense_mask, prob, torch.ones_like(prob))

        if log_likelihood:
            return torch.log(likelihood + 1e-8)

        return likelihood

    def postprocess_belief_particles(
        self, 
        prior_belief_particles: torch.Tensor, 
        action: torch.Tensor,
        observation: torch.Tensor,
        belief_particles: torch.Tensor,
    ):
        return _postprocess_belief_particles(
            prior_belief_particles,
            action,
            observation,
            belief_particles,
            self.rock_positions.shape[0],
            self.rock_positions.unsqueeze(0).expand(prior_belief_particles.shape[0], -1, -1),
        )

    def _likelihood(
        self,
        observation: torch.Tensor,    # (B, 2), values in {0, 1, 2}
        next_state: torch.Tensor,     # (B, 4 + m)
        action: torch.Tensor          # (B, 2), per-agent action ID
    ) -> torch.Tensor:
        """
        Computes p(o | s', a) for multi-agent RockSample.

        Returns:
            likelihood: (B,) - probability of each observation under s', a
        """
        B = next_state.shape[0]
        m = self.rock_positions.shape[0]
        half_eff = 20.0
        TERMINAL = -1

        device = next_state.device
        agent_pos = next_state[:, :4].view(B, 2, 2)      # (B, 2, 2)
        rock_state = next_state[:, 4:]                   # (B, m)

        # Split observation per agent
        obs = observation                                # (B, 2)
        a = action                                       # (B, 2)

        likelihoods = torch.ones((B,), dtype=torch.float32, device=device)

        for i in range(2):
            ai = a[:, i]             # (B,)
            oi = obs[:, i]           # (B,)
            pos = agent_pos[:, i]    # (B, 2)
            is_active = pos[:, 0] != TERMINAL

            # Case 1: non-sense actions (<= SAMPLE)
            non_sense = (ai <= 4) & is_active
            prob_none = (oi == 0).float()
            likelihoods *= torch.where(non_sense, prob_none, torch.ones_like(prob_none))

            # Case 2: sense actions
            sense = (ai > 4) & is_active & ((oi == 1) | (oi == 2))
            rock_ids = (ai - 5).clamp(min=0, max=m-1)  # (B,)

            # Rock positions
            rock_pos = self.rock_positions[rock_ids]  # (B, 2)
            dist = torch.norm(pos - rock_pos.to(pos), dim=1)  # (B,)
            eff = 0.5 * (1 + torch.pow(2, -dist / half_eff))   # (B,)

            # Extract rock truth (0=BAD, 1=GOOD)
            truth = torch.gather(rock_state, 1, rock_ids.unsqueeze(1)).squeeze(1)  # (B,)

            obs_is_good = (oi == 2).float()
            truth_is_good = (truth == 1).float()

            match = (obs_is_good == truth_is_good).float()  # 1 if match, else 0
            prob = match * eff + (1 - match) * (1 - eff)    # (B,)

            likelihoods *= torch.where(sense, prob, torch.ones_like(prob))

            # Case 3: invalid observations
            invalid_obs = is_active & ((oi != 0) & (oi != 1) & (oi != 2))
            likelihoods *= torch.where(invalid_obs, torch.zeros_like(likelihoods), torch.ones_like(likelihoods))

        return likelihoods

    def action_repr(self, action: torch.Tensor) -> str:
        act = action.to(dtype=torch.int64).view(-1)
        decoded = decode_action_ids(act, num_rocks=self._num_rocks)  # shape (1, 2)

        # Define mapping
        base_actions = ["EAST", "NORTH", "SOUTH", "WEST", "SAMPLE"]
        rock_actions = [f"SCAN_ROCK_{i}" for i in range(self._num_rocks)]
        action_map = base_actions + rock_actions

        a0 = decoded[0][0].item()
        a1 = decoded[0][1].item()

        def action_to_str(a):
            if a < len(base_actions):
                return action_map[a]
            else:
                rock_idx = a - len(base_actions)
                return f"SCAN_ROCK_{rock_idx}"

        a0_str = action_to_str(a0)
        a1_str = action_to_str(a1)

        return f"action ID: {act.item()}, agent0: {a0_str}, agent1: {a1_str}"        

    def is_goal(self, state: torch.Tensor) -> torch.Tensor: 
        if state[0][1] == -1 and state[0][3] == -1:
            return torch.ones((1,), dtype=torch.bool, device=state.device)
        return torch.zeros((1,), dtype=torch.bool, device=state.device)

    def get_info(self):
        return (
            f"num_good_rocks_initial = {self._num_good_rocks_initial}, "
            f"num_good_rocks = {self._num_good_rocks_current}, "
            f"num_bad_rock_samplings = {self._num_bad_rock_samplings}"
        )        

    def plot(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor,
        observation: torch.Tensor, 
        belief: torch.Tensor
    ):
        if self.headless:
            return

        grid_h = self._map_size
        grid_w = self._map_size
        m = self.rock_positions.shape[0]

        # Unpack state
        agent_pos = state[0, :4].view(2, 2).int().cpu()  # shape (2, 2)
        rock_state = state[0, 4:4 + m].int().cpu()

        checked_rock_idxs = [None, None]
        if action is not None:
            action = action.cpu()
            if action.dim() == 2 and action.shape == (1, 1):
                action = action.view(-1)  # Shape (1,)
            if action.dim() == 0:
                action = action.unsqueeze(0)  # Shape (1,)
            decoded = decode_action_ids(action, num_rocks=m)[0]  # Shape (2,)
            action = decoded
            #checked_rock_idxs = [a - 5 if a >= 5 else None for a in action]
            checked_rock_idxs = []
            for i in range(2):
                a = action[i].item()
                ax, _ = agent_pos[i].tolist()
                if ax != -1 and a >= 5:
                    checked_rock_idxs.append(a - 5)
                else:
                    checked_rock_idxs.append(None)

        # Create figure and axis only once
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(grid_w, grid_h))
            self.ax.set_xlim(-0.5, grid_w + 0.5)
            self.ax.set_ylim(-0.5, grid_h - 0.5)
            self.ax.set_aspect('equal')
            self.ax.set_xticks(range(grid_w + 1))
            self.ax.set_yticks(range(grid_h))
            self.ax.grid(True, which='both')
            plt.ion()
            plt.show()

        self.ax.clear()  # Clear previous content

        # Reset axis settings after clearing
        self.ax.set_xlim(-0.5, grid_w + 0.5)
        self.ax.set_ylim(-0.5, grid_h - 0.5)
        self.ax.set_aspect('equal')
        self.ax.set_xticks(range(grid_w + 1))
        self.ax.set_yticks(range(grid_h))
        self.ax.grid(True, which='both')

        # Draw rocks
        for i, (x, y) in enumerate(self.rock_positions.cpu().tolist()):
            status = rock_state[i].item()
            if i in checked_rock_idxs:
                color = 'yellow'
            elif status == 1:
                color = 'green'
            elif status == 0:
                color = 'red'
            else:
                color = 'gray'

            self.ax.add_patch(plt.Circle((x, grid_h - 1 - y), 0.3, color=color, alpha=0.8))
            self.ax.text(x, grid_h - 1 - y, f'R{i}', ha='center', va='center', fontsize=8, color='black')

        # Draw agents
        for i in range(2):
            ax, ay = agent_pos[i].tolist()
            if ax == -1:
                continue  # Skip if terminal
            self.ax.add_patch(plt.Rectangle((ax - 0.4, grid_h - 1 - ay - 0.4), 0.8, 0.8, color='blue'))
            self.ax.text(ax, grid_h - 1 - ay, f'A{i}', ha='center', va='center', fontsize=12, color='white')

            if action is not None and action[i] == 4:
                self.ax.text(
                    ax, grid_h - 1 - ay,
                    "SAMPLE",
                    ha='center', va='center',
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", fc="orange", ec="black", lw=1),
                    color='black'
                )

        self.ax.set_title("Multi-Agent RockSample")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        if belief is not None:
            print("\nrock beliefs:")
            rock_bel = belief[:, 4:4+self.rock_positions.shape[0]].float().mean(dim=0).cpu()
            scanned = belief[:, 4+self.rock_positions.shape[0]:].float().mean(dim=0).cpu()
            for i in range(rock_bel.shape[0]):
                print(f"rock {i}: ({rock_bel[i].item():.2f}, {scanned[i].item():.2f})")
