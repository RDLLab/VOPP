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


ACTIONS = torch.tensor([
    [-1,  0],  # N
    [-1,  1],  # NE
    [ 0,  1],  # E
    [ 1,  1],  # SE
    [ 1,  0],  # S
    [ 1, -1],  # SW
    [ 0, -1],  # W
    [-1, -1],  # NW
    [ 0,  0],  # STAY
], dtype=torch.int64)

STEP_NOISE = 0.03
#OBS_NOISE = 0.03
OBS_NOISE = 0.03

def generate_known_map(device):
    grid_size = 13
    num_obstacles = 31
    grid = torch.zeros((grid_size, grid_size), dtype=torch.int32, device=device)

    wall_x = 6
    grid[wall_x, :] = 1  # fully closed wall

    goal_pos = (12, 6)
    grid[goal_pos] = 2

    # Forbidden: full wall and goal cell
    forbidden = set([(wall_x, y) for y in range(grid_size)])
    forbidden.add(goal_pos)

    all_positions = [(x, y) for x in range(grid_size) for y in range(grid_size)]
    valid_positions = [pos for pos in all_positions if pos not in forbidden and grid[pos] == 0]

    obstacle_indices = torch.randperm(len(valid_positions))[:num_obstacles]
    obstacle_coords = torch.tensor([valid_positions[i] for i in obstacle_indices], dtype=torch.long, device=device)
    grid[obstacle_coords[:, 0], obstacle_coords[:, 1]] = 1

    return grid

def sample_maps_from_known(base_map: torch.Tensor, N: int) -> torch.Tensor:
    """
    Sample N maps by:
    - Adding an opening at y=3 or y=9 on row x=6 (one per map).
    - Sampling obstacles in remaining free cells (prob=0.1).
    - Placing a robot in a random free cell in row x=0.
    
    Returns:
        Tensor of shape (N, 13, 15), where:
        - [:, :, :13] is the map
        - [:, 0, 13] is the robot x position (always 0)
        - [:, 0, 14] is the robot y position
    """
    device = base_map.device
    grid_size = base_map.shape[-1]
    maps = base_map.unsqueeze(0).expand(N, -1, -1).clone()

    # Step 1: Randomly assign gate opening at y=3 or y=9
    opening_choices = torch.randint(0, 2, (N,), device=device)  # 0 or 1
    opening_ys = torch.where(opening_choices == 0, 3, 9)

    # Step 2: Build fillable mask (free cells not including goal)
    fillable_mask = (base_map == 0)
    fillable_mask[12, 6] = False  # don't overwrite goal

    per_map_fillable = fillable_mask.unsqueeze(0).expand(N, -1, -1).clone()

    # Step 3: Sample new obstacles (may later be overwritten by gate clearing)
    rand_vals = torch.rand((N, grid_size, grid_size), device=device)
    sampled_obstacles = (rand_vals < 0.1) & per_map_fillable
    maps[sampled_obstacles] = 1

    # Step 4: Insert the gate and its vertical corridor (overwriting sampled obstacles if needed)
    i = torch.arange(N, device=device)
    y = opening_ys
    maps[i, 6, y] = 0  # gate
    maps[i, 5, y] = 0  # above
    maps[i, 7, y] = 0  # below

    # Step 5: Place robot in a random free cell in row 0
    row0_free_mask = (maps[:, 0, :] == 0)
    rand_scores = torch.rand(N, grid_size, device=device)
    rand_scores[~row0_free_mask] = -float('inf')
    robot_y = rand_scores.argmax(dim=1)
    robot_x = torch.zeros_like(robot_y)

    # Mark robot on the map (optional visual marker)
    #maps[i, robot_x, robot_y] = 3

    # Step 6: Construct final state tensor with robot position stored separately
    states = torch.zeros(N, 13, 15, dtype=maps.dtype, device=device)
    states[:, :, :13] = maps
    states[:, 0, 13] = robot_x
    states[:, 0, 14] = robot_y

    return states

def observation_to_id(obs: torch.Tensor) -> torch.Tensor:
    """
    Converts an 8-bit binary observation (B, 8) to an integer ID (B,) in [0, 255].
    Assumes obs[..., 0] is the most significant bit (MSB), obs[..., 7] is the LSB.
    """
    powers = 2 ** torch.arange(7, -1, -1, device=obs.device)
    return (obs.to(torch.int64) * powers).sum(dim=-1)

def id_to_observation(obs_id: torch.Tensor) -> torch.Tensor:
    """
    Converts an integer ID (B,) in [0, 255] back to 8-bit binary observation (B, 8).
    """
    obs_id = obs_id.to(torch.uint8)
    return ((obs_id.unsqueeze(-1) >> torch.arange(7, -1, -1, device=obs_id.device)) & 1).to(torch.int32)

def plot_map(map_tensor: torch.Tensor, ax=None, title=None):
    """
    Plot a single map with shape (13, 15), where the robot position is encoded
    in map_tensor[0, 13] (x) and map_tensor[0, 14] (y). The base map is in [:, :13].

    Args:
        map_tensor (torch.Tensor): (13, 15) tensor with values:
                                   0=free, 1=obstacle, 2=goal
                                   robot position is in [:, 13:] columns
        ax (matplotlib.axes.Axes): Optional matplotlib axis to plot into
        title (str): Optional title for the plot
    """
    # Copy base map and insert robot marker
    grid = map_tensor[:, :13].clone()
    x = int(map_tensor[0, 13].item())
    y = int(map_tensor[0, 14].item())
    if 0 <= x < 13 and 0 <= y < 13:
        grid[x, y] = 3  # Mark robot position

    # Define color map
    cmap = mcolors.ListedColormap(['white', 'black', 'green', 'red'])  # free, wall, goal, robot
    bounds = [0, 1, 2, 3, 4]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    ax.imshow(grid.cpu().numpy(), cmap=cmap, norm=norm)

    ax.set_xticks(range(13))
    ax.set_yticks(range(13))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([x - 0.5 for x in range(1, 13)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, 13)], minor=True)
    ax.grid(which='minor', color='lightgray', linestyle='-', linewidth=0.5)

    if title:
        ax.set_title(title)

    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig('unc_nav.png')
    plt.close()

def step(s: torch.Tensor, a: torch.Tensor, all_actions: torch.Tensor):    
    B, H, W = s.shape
    assert H == 13 and W == 15

    device = s.device
    s_next = s.clone()
    map_grid = s[:, :, :13]

    # Get robot positions
    x = s[:, 0, 13].to(torch.long)  # (B,)
    y = s[:, 0, 14].to(torch.long)  # (B,)

    # Step noise
    fails = torch.rand(B, device=device) < STEP_NOISE
    effective_a = torch.where(fails, torch.full_like(a, 8), a)  # fallback to STAY

    dxdy = all_actions[effective_a]  # (B, 2)
    new_x = x + dxdy[:, 0]
    new_y = y + dxdy[:, 1]

    in_bounds = (new_x >= 0) & (new_x < 13) & (new_y >= 0) & (new_y < 13)
    target_val = torch.full((B,), 1, dtype=torch.int32, device=device)
    target_val[in_bounds] = map_grid[torch.arange(B, device=device)[in_bounds], new_x[in_bounds], new_y[in_bounds]]
    move_valid = in_bounds & (target_val != 1)

    # Reward computation
    reward = torch.full((B,), -0.1, dtype=torch.float32, device=device)
    reward[a == 8] = -0.2
    reward[~move_valid & (a != 8)] = -1.0

    # Update robot position
    final_x = torch.where(move_valid, new_x, x)
    final_y = torch.where(move_valid, new_y, y)

    s_next[:, 0, 13] = final_x
    s_next[:, 0, 14] = final_y

    # Check goal
    goal_reached = map_grid[torch.arange(B, device=device), final_x, final_y] == 2
    reward[goal_reached] = 20.0
    done = goal_reached

    # Observation (8 directions)
    OBS_DIRS = all_actions[:8]
    dx = OBS_DIRS[:, 0].view(1, 8)
    dy = OBS_DIRS[:, 1].view(1, 8)

    x_exp = final_x.view(-1, 1) + dx
    y_exp = final_y.view(-1, 1) + dy
    valid = (x_exp >= 0) & (x_exp < 13) & (y_exp >= 0) & (y_exp < 13)

    obs = torch.zeros((B, 8), dtype=torch.bool, device=device)
    valid_idx = valid.nonzero(as_tuple=False)

    obs_vals = torch.zeros(B, 8, dtype=torch.bool, device=device)
    obs_vals[valid] = map_grid[
        valid_idx[:, 0],
        x_exp[valid],
        y_exp[valid]
    ] == 1

    flip = torch.rand(B, 8, device=device) < OBS_NOISE
    obs = (obs_vals ^ flip).to(torch.int64)

    return s_next, reward, done, obs

def _likelihood(s: torch.Tensor, obs: torch.Tensor, all_actions: torch.Tensor) -> torch.Tensor:
    """
    Compute p(o | s) for a batch of states and observations.
    Args:
        s (torch.Tensor): (B, 13, 15) tensor, where s[:, :, :13] is the map,
                          and s[:, 0, 13:15] contains robot x and y position.
        obs (torch.Tensor): (B, 8) tensor of 0/1 observations.        
    Returns:
        torch.Tensor: (B,) log-likelihoods p(o | s) as float32 tensor.
    """
    B = s.shape[0]
    device = s.device
    map_grid = s[:, :, :13]
    x = s[:, 0, 13].to(torch.long)  # (B,)
    y = s[:, 0, 14].to(torch.long)  # (B,)

    # Define 8 directions
    DIRS = all_actions[:8]

    dx = DIRS[:, 0].view(1, 8)  # (1, 8)
    dy = DIRS[:, 1].view(1, 8)  # (1, 8)

    x_exp = x.view(-1, 1) + dx  # (B, 8)
    y_exp = y.view(-1, 1) + dy  # (B, 8)

    valid = (x_exp >= 0) & (x_exp < 13) & (y_exp >= 0) & (y_exp < 13)  # (B, 8)

    ground_truth_obs = torch.zeros(B, 8, dtype=torch.float32, device=device)
    idx = valid.nonzero(as_tuple=False)
    ground_truth_obs[valid] = (map_grid[idx[:, 0], x_exp[valid], y_exp[valid]] == 1).float()    

    # Compute log-likelihood under independent Bernoulli with flip probability
    p = OBS_NOISE
    probs = ground_truth_obs * (1 - p) + (1 - ground_truth_obs) * p  # P(o_i | s)    
    log_probs = obs.float() * torch.log(probs + 1e-8) + (1 - obs.float()) * torch.log(1 - probs + 1e-8)
    return log_probs.sum(dim=1)  # (B,)

def _heuristic(states: torch.Tensor, discount_factor: float = 0.95) -> torch.Tensor:
    B = states.shape[0]
    device = states.device
    H = W = 13
    INF = 10**6

    # Extract map and robot positions
    grid = states[:, :, :13]  # (B, 13, 13)
    robot_x = states[:, 0, 13].long()
    robot_y = states[:, 0, 14].long()

    # Goal
    goal_x, goal_y = 12, 6

    # Distance grid
    dist = torch.full((B, 1, H, W), INF, dtype=torch.int32, device=device)

    # Frontier initialization (start from the goal)
    frontier = torch.zeros((B, 1, H, W), dtype=torch.float32, device=device)
    frontier[:, 0, goal_x, goal_y] = 1.0
    dist[:, 0, goal_x, goal_y] = 0

    # Free cell mask
    free_mask = (grid != 1).unsqueeze(1)  # (B,1,H,W)

    # 3x3 kernel for 8-connected neighbors
    kernel = torch.tensor(
        [[1, 1, 1],
         [1, 0, 1],
         [1, 1, 1]],
        dtype=torch.float32, device=device
    ).view(1, 1, 3, 3)

    d = 0
    while frontier.any():
        d += 1
        # Find neighbors of current frontier
        neigh = F.conv2d(frontier, kernel, padding=1)
        # New frontier: neighbors that are free and not visited
        new_frontier = (neigh > 0) & free_mask & (dist == INF)
        dist[new_frontier] = d
        frontier = new_frontier.float()

    # Extract distance at robot position
    robot_nodes = dist[torch.arange(B, device=device), 0, robot_x, robot_y]

    # Compute heuristic values
    gamma = torch.tensor(discount_factor, device=device)
    d_float = robot_nodes.float()
    motion_cost = (-0.1) * (1 - gamma.pow(d_float)) / (1 - gamma)
    value = motion_cost + gamma.pow(d_float) * 20.0
    value[robot_nodes >= INF] = -100.0  # unreachable
    return value


def _heuristic_old(states: torch.Tensor, discount_factor: float = 0.95) -> torch.Tensor:
    """
    Compute a discounted reward heuristic for reaching the goal on a batch of maps using 8-connected BFS.

    Args:
        states (Tensor): (B, 13, 15) tensor where [:, :, :13] is the map,
                         [:, 0, 13] is robot x, [:, 0, 14] is robot y
        discount_factor (float): γ discount factor applied per step

    Returns:
        Tensor: (B,) discounted reward estimate for each state
    """
    B, H, W = states.shape
    #assert H == 13 and W == 15
    device = states.device

    grid = states[:, :, :13]  # (B, 13, 13)
    robot_x = states[:, 0, 13].long()  # (B,)
    robot_y = states[:, 0, 14].long()  # (B,)

    goal_x, goal_y = 12, 6

    neighbors = torch.tensor([
        [-1,  0],  # N
        [-1,  1],  # NE
        [ 0,  1],  # E
        [ 1,  1],  # SE
        [ 1,  0],  # S
        [ 1, -1],  # SW
        [ 0, -1],  # W
        [-1, -1],  # NW
    ], device=device)

    visited = torch.zeros(B, 13, 13, dtype=torch.bool, device=device)
    distance = torch.full((B, 13, 13), -1, dtype=torch.int32, device=device)

    current_front = torch.zeros(B, 13, 13, dtype=torch.bool, device=device)
    current_front[:, goal_x, goal_y] = True
    visited[:, goal_x, goal_y] = True
    distance[:, goal_x, goal_y] = 0

    d = 0
    while current_front.any():
        d += 1
        next_front = torch.zeros_like(current_front)

        for dx, dy in neighbors:
            shifted = torch.roll(current_front, shifts=(dx, dy), dims=(1, 2))

            # Set borders to False (prevent wraparound)
            if dx == -1:
                shifted[:, -1, :] = False
            elif dx == 1:
                shifted[:, 0, :] = False
            if dy == -1:
                shifted[:, :, -1] = False
            elif dy == 1:
                shifted[:, :, 0] = False

            # Valid new front: not visited and not obstacle
            valid = (~visited) & (grid != 1) & shifted
            distance[valid] = d
            visited |= valid
            next_front |= valid

        current_front = next_front

    # Path length from robot to goal
    path_len = distance[torch.arange(B), robot_x, robot_y]  # (B,)
    unreachable = path_len == -1

    d_float = path_len.float()
    gamma = torch.tensor(discount_factor, device=device)
    motion_cost = (-0.1) * (1 - gamma.pow(d_float)) / (1 - gamma)
    goal_reward = 20.0 * gamma.pow(d_float)
    value = motion_cost + goal_reward

    #value[unreachable] = -100.0
    return value  # (B,)


class UncNavigation(GenerativeModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gamma = kwargs.get('args_cli').discount_factor        
        self._action_ranges = [[i for i in range(9)]]
        self._device = kwargs.get('args_cli').device        
        self.step_penalty = -1.0
        self.goal_reward = 100.0
        self.all_actions = ACTIONS.to(self._device)
        self.headless = kwargs.get('args_cli').headless
        self._fig, self._ax = None, None
        if self.role == 'planning':
            self.exec_env = kwargs.get('exec_env', None)   

        self._num_actions = 9

    @property
    def num_actions(self) -> int:
        return self._num_actions

    def get_shared_data(self):
        return {'known_map': self.known_map}

    def reset(self):
        if self.role == 'exec':
            self.known_map = generate_known_map(self._device)
        else:
            data = self.exec_env.get_shared_data()
            self.known_map = data['known_map'].to(device=self._device)

    def sample_initial_belief(self, num_samples: int = 1) -> torch.Tensor:
        initial_belief_particles = sample_maps_from_known(
            self.known_map, 
            num_samples
        ).view(-1, 13*15)
        return initial_belief_particles

    def sample(self, state: torch.Tensor, action: torch.Tensor, **kwargs) -> dict:        
        next_state, reward, done, obs = step(
            state.view(-1, 13, 15), 
            action, 
            self.all_actions
        )

        return {
            'next_state': next_state.view(-1, 13*15),
            'observation': observation_to_id(obs),
            'reward': reward,
            'terminal': done,
        }

    def heuristics(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor, 
        current_nodes: Optional[torch.Tensor] = None
    ) -> torch.Tensor:        
        heuristic_values = _heuristic(state.view(-1, 13, 15), discount_factor=self.gamma)
        return heuristic_values

    def likelihood(
        self, 
        observation: torch.Tensor,
        states: torch.Tensor, 
        next_state: torch.Tensor, 
        action: torch.Tensor, 
        log_likelihood: bool = False,
        is_encoded_observation: bool = True,
    ) -> torch.Tensor:        
        log_probs = _likelihood(next_state.view(-1, 13, 15), id_to_observation(observation), self.all_actions)
        if log_likelihood:
            return log_probs
        return log_probs.exp()

    def is_goal(self, state: torch.Tensor) -> torch.Tensor: 
        s = state.view(-1, 13, 15).to(dtype=torch.int32)       
        pos_x = s[0, 0, 13]
        pos_y = s[0, 0, 14]
        return (pos_x == 12) & (pos_y == 6)

    def _plot(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor,
        observation: torch.Tensor, 
        belief: torch.Tensor
    ):
        map_tensor = state[0].view(13, 15)
        grid = map_tensor[:, :13].clone()
        x = int(map_tensor[0, 13].item())
        y = int(map_tensor[0, 14].item())
        if 0 <= x < 13 and 0 <= y < 13:
            grid[x, y] = 3  # Mark robot position

        # Define color map
        cmap = mcolors.ListedColormap(['white', 'black', 'green', 'red'])  # free, wall, goal, robot
        bounds = [0, 1, 2, 3, 4]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        # Enable interactive mode
        plt.ion()

        if self._fig is None or self._ax is None:
            self._fig, self._ax = plt.subplots(figsize=(5, 5))

        ax = self._ax
        ax.clear()
        ax.imshow(grid.cpu().numpy(), cmap=cmap, norm=norm)

        ax.set_xticks(range(13))
        ax.set_yticks(range(13))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([x - 0.5 for x in range(1, 13)], minor=True)
        ax.set_yticks([y - 0.5 for y in range(1, 13)], minor=True)
        ax.grid(which='minor', color='lightgray', linestyle='-', linewidth=0.5)
        ax.set_aspect('equal')

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    def _postprocess_belief_particles(self, belief_particles: torch.Tensor) -> torch.Tensor:
        # TEST ME
        alpha = 0.05
        N = belief_particles.shape[0]
        N_prior = int(alpha * N)
        N_posterior = N - N_prior
        prior_particles = sample_maps_from_known(self.known_map, N_prior).view(-1, 13*15)
        return torch.cat([belief_particles[:N_posterior], prior_particles], dim=0)

    def postprocess_belief_particles(
        self, 
        prior_belief_particles: torch.Tensor,
        action: torch.Tensor,
        observation: torch.Tensor, 
        belief_particles: torch.Tensor
    ) -> torch.Tensor:
        """
        Adds flipping noise only to cells where all particles agree (free or occupied),
        excluding goal cells and known occupied map cells. Applies noise independently per particle.

        Args:
            belief_particles (Tensor): shape (N, 13 * 15), flattened belief particles.

        Returns:
            Tensor: updated belief particles, same shape.
        """
        flip_prob = 0.001  # Careful: still moderate value, apply only to unanimous cells

        particles = belief_particles.view(-1, 13, 15)  # (N, 13, 15)
        N, H, W = particles.shape
        device = particles.device

        # Split map and robot parts
        map_part = particles[:, :, :13]        # (N, 13, 13)
        robot_part = particles[:, :, 13:]      # (N, 13, 2)

        # Detect which cells are all 0 or all 1 across particles
        #all_free = (map_part == 0).all(dim=0)   # (13, 13)
        #all_occ  = (map_part == 1).all(dim=0)   # (13, 13)

        # Original known occupied mask
        known_occupied_mask = self.known_map == 1  # (13, 13)

        # Override known gate cells to not be treated as known
        known_occupied_mask = known_occupied_mask.clone()
        known_occupied_mask[6, 3] = False
        known_occupied_mask[6, 9] = False

        # Candidate: unanimous and not known occupied
        #unanimous_mask = (all_free | all_occ) & ~known_occupied_mask  # (13, 13)
        unanimous_mask = ~known_occupied_mask  # (13, 13)

        # Broadcast this mask to all particles
        broadcast_mask = unanimous_mask.unsqueeze(0).expand(N, -1, -1)  # (N, 13, 13)

        # Now flip per-particle per-cell with flip_prob
        rand_mask = torch.rand((N, 13, 13), device=device) < flip_prob  # (N, 13, 13)
        final_flip_mask = broadcast_mask & rand_mask  # Only flip in unanimous regions

        # Flip values (0 <-> 1), only if they are 0 or 1
        flip_target = (map_part == 0) | (map_part == 1)
        map_flipped = map_part.clone()
        map_flipped[final_flip_mask & flip_target] = 1 - map_flipped[final_flip_mask & flip_target]

        # Recombine
        new_particles = torch.cat([map_flipped, robot_part], dim=2)
        return new_particles.view(N, -1)

    def plot(self, state: torch.Tensor, action: torch.Tensor, observation: torch.Tensor, belief: torch.Tensor):
        """
        Visualize the true state and combined belief map with:
        - grayscale darkness for obstacle belief
        - red overlay for robot belief

        Args:
            state (Tensor): shape (1, 13, 15)
            action (Tensor): unused
            belief (Tensor): shape (B, 13, 15), with [:, :, :13] = map, [:, 0, 13:15] = robot pos
        """
        if self.headless:
            return

        map_tensor = state[0].view(13, 15).clone()
        _belief = belief.view(-1, 13, 15).clone()
        grid = map_tensor[:, :13].clone()
        x = int(map_tensor[0, 13].item())
        y = int(map_tensor[0, 14].item())
        if 0 <= x < 13 and 0 <= y < 13:
            grid[x, y] = 3  # mark robot

        # Color map for true state
        cmap = mcolors.ListedColormap(['white', 'black', 'green', 'red'])
        bounds = [0, 1, 2, 3, 4]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        # Init figure
        plt.ion()
        if not hasattr(self, "_fig") or self._fig is None or not hasattr(self, "_axs") or self._axs is None:
            self._fig, self._axs = plt.subplots(1, 2, figsize=(10, 5))
        axs = self._axs
        for ax in axs:
            ax.clear()

        # Plot true state
        axs[0].imshow(grid.cpu().numpy(), cmap=cmap, norm=norm)
        axs[0].set_title("True State")

        # --------------------------
        # Combined belief rendering
        # --------------------------
        B = _belief.shape[0]
        belief_maps = _belief[:, :, :13]
        robot_x = _belief[:, 0, 13].long()
        robot_y = _belief[:, 0, 14].long()

        # Obstacle belief (grayscale darkness)
        obstacle_belief = (belief_maps == 1).float().mean(dim=0).cpu().numpy()  # (13, 13)

        # Robot belief (red tint)
        robot_belief = torch.zeros(13, 13, device=_belief.device)
        robot_belief.index_put_((robot_x, robot_y), torch.ones(B, device=_belief.device), accumulate=True)
        robot_belief /= B
        robot_belief = robot_belief.cpu().numpy()  # (13, 13)

        # Step 1: White base
        rgb = np.ones((13, 13, 3), dtype=np.float32)

        # Step 2: Grayscale darkening from obstacle belief
        rgb -= obstacle_belief[:, :, None]  # subtract equally from R,G,B

        # Step 3: Red tint for robot belief
        red_intensity = robot_belief.clip(0, 1)
        rgb[:, :, 1] -= red_intensity  # reduce green
        rgb[:, :, 2] -= red_intensity  # reduce blue

        rgb = np.clip(rgb, 0.0, 1.0)

        axs[1].imshow(rgb)
        axs[1].set_title("Belief (Obstacle + Robot)")

        for ax in axs:
            ax.set_xticks(range(13))
            ax.set_yticks(range(13))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([i - 0.5 for i in range(1, 13)], minor=True)
            ax.set_yticks([i - 0.5 for i in range(1, 13)], minor=True)
            ax.grid(which='minor', color='lightgray', linestyle='-', linewidth=0.5)
            ax.set_aspect('equal')

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()  
