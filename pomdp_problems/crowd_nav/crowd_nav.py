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
from matplotlib.patches import Circle

from typing import Optional

#from .pedestrian_model import fused_step_and_count_host, step_and_count_split_host, MAP_SIZE, device
#from .ped_sim import PedSim

ACTIONS = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
#SPEED_MIN, SPEED_MAX = 0.7, 0.7  # Speed range
#CURIOUS_SPEED, SHY_SPEED = 0.3, 0.8
CURIOUS_SPEED = 0.3 
SHY_SPEED = 0.8
YELL_SPEED = 2.0

'''def step_pedestrians(
    state: torch.Tensor,
    action: torch.Tensor, 
    map_size_x: float,
    map_size_y: float,
    noise_std: float, 
    move_prob: float = 1.0,
    curiosity_distance: float = 3.5,
    drift_speed: float = 0.0,
) -> torch.Tensor:
    """
    Updates pedestrian positions based on curiosity and robot action.
    If the robot yells (action == 4), all peds within curiosity_distance move away using SHY_SPEED.
    """

    B, D = state.shape
    assert D % 5 == 0
    N = D // 5 - 1
    state = state.view(B, 1 + N, 5)

    # Extract components
    robot_pos = state[:, 0, :2]          # (B, 2)
    ped_pos = state[:, 1:, :2]           # (B, N, 2)
    ped_speed = state[:, 1:, 2:3]        # (B, N, 1)
    ped_curiosity = state[:, 1:, 3:4]    # (B, N, 1)

    move_mask = (torch.rand(B, N, 1, device=state.device) < move_prob)  # (B, N, 1)

    rel = robot_pos.unsqueeze(1) - ped_pos     # (B, N, 2)
    dist = rel.norm(dim=-1, keepdim=True)      # (B, N, 1)
    dir_to_robot = rel / (dist + 1e-6)         # (B, N, 2)

    curiosity_mask = (dist <= curiosity_distance).float()  # (B, N, 1)

    is_yell = (action == 4).view(B, 1, 1)  # (B, 1, 1)

    # Direction: either based on curiosity or yell
    base_sign = 2 * ped_curiosity - 1
    dir_sign = torch.where(is_yell, -1.0, base_sign)  # (B, N, 1)
    move_dir = curiosity_mask * dir_sign * dir_to_robot  # (B, N, 2)

    # Speed override: if yelling and within curiosity_distance, use SHY_SPEED
    shy_speed = torch.full_like(ped_speed, YELL_SPEED)
    use_shy_speed = is_yell & (curiosity_mask > 0.0)
    effective_speed = torch.where(use_shy_speed, shy_speed, ped_speed)  # (B, N, 1)

    base_motion = effective_speed.expand(-1, -1, 2) * move_dir
    noise = torch.randn_like(base_motion) * noise_std
    drift_motion = torch.tensor([-drift_speed, 0.0], device=state.device).view(1, 1, 2).expand(B, N, 2)

    delta = base_motion * move_mask + noise + drift_motion
    next_pos = ped_pos + delta

    # Clamp: x ∈ [0, map_size_x], y ∈ [-map_size_y/2, +map_size_y/2]
    y_half = map_size_y / 2.0
    next_pos[..., 0] = torch.clamp(next_pos[..., 0], 0.0, map_size_x)         # x (forward)
    next_pos[..., 1] = torch.clamp(next_pos[..., 1], -y_half, +y_half)        # y (sideways)

    next_state = state.clone()
    next_state[:, 1:, :2] = next_pos
    return next_state.view(B, -1)'''

def step_pedestrians(
    state: torch.Tensor,
    action: torch.Tensor, 
    map_size_x: float,
    map_size_y: float,
    noise_std: float, 
    move_prob: float = 1.0,
    curiosity_distance: float = 3.5,
    drift_speed: float = 0.0,
) -> torch.Tensor:
    """
    Updates pedestrian positions based on curiosity and robot action.
    If the robot yells (action == 4), all peds within curiosity_distance move away using YELL_SPEED,
    and always move regardless of move_prob.
    """
    B, D = state.shape
    assert D % 5 == 0
    N = D // 5 - 1
    state = state.view(B, 1 + N, 5)

    # Extract components
    robot_pos = state[:, 0, :2]          # (B, 2)
    ped_pos = state[:, 1:, :2]           # (B, N, 2)
    ped_speed = state[:, 1:, 2:3]        # (B, N, 1)
    ped_curiosity = state[:, 1:, 3:4]    # (B, N, 1)

    # Relative position to robot
    rel = robot_pos.unsqueeze(1) - ped_pos     # (B, N, 2)
    dist = rel.norm(dim=-1, keepdim=True)      # (B, N, 1)
    dir_to_robot = rel / (dist + 1e-6)         # (B, N, 2)

    curiosity_mask = (dist <= curiosity_distance).float()  # (B, N, 1)
    curiosity_mask_bool = (dist <= curiosity_distance)     # (B, N, 1)

    is_yell = (action == 4).view(B, 1, 1)  # (B, 1, 1)

    # Direction: toward if curious, away if not
    base_sign = 2 * ped_curiosity - 1      # 1 if curious, -1 if not
    dir_sign = torch.where(is_yell, -1.0, base_sign)  # Invert if YELL
    move_dir = curiosity_mask * dir_sign * dir_to_robot  # (B, N, 2)

    # Speed override: use YELL_SPEED if yelling and within curiosity distance
    shy_speed = torch.full_like(ped_speed, YELL_SPEED)
    use_shy_speed = is_yell & curiosity_mask_bool
    effective_speed = torch.where(use_shy_speed, shy_speed, ped_speed)  # (B, N, 1)

    # === Move mask: override with force move if yelling and close ===
    move_mask = (torch.rand(B, N, 1, device=state.device) < move_prob)  # base mask
    force_move_mask = is_yell & curiosity_mask_bool
    move_mask = move_mask | force_move_mask  # always move if yelling and within range

    # === Compute final motion ===
    base_motion = effective_speed.expand(-1, -1, 2) * move_dir           # (B, N, 2)
    noise = torch.randn_like(base_motion) * noise_std                   # (B, N, 2)
    drift_motion = torch.tensor([-drift_speed, 0.0], device=state.device).view(1, 1, 2).expand(B, N, 2)

    delta = base_motion * move_mask + noise + drift_motion
    next_pos = ped_pos + delta

    # Clamp: x ∈ [0, map_size_x], y ∈ [-map_size_y/2, +map_size_y/2]
    y_half = map_size_y / 2.0
    next_pos[..., 0] = torch.clamp(next_pos[..., 0], 0.0, map_size_x)         # x (forward)
    next_pos[..., 1] = torch.clamp(next_pos[..., 1], -y_half, +y_half)        # y (sideways)

    # Final state
    next_state = state.clone()
    next_state[:, 1:, :2] = next_pos
    return next_state.view(B, -1)




'''def observe(prev_state: torch.Tensor, next_state: torch.Tensor, num_near_peds: int) -> torch.Tensor:
    """
    Computes discrete observation ID for each sample based on whether each of the
    `num_near_peds` pedestrians moved towards or away from the robot.

    Pedestrian matching is done via ped_id to ensure correct alignment.

    Args:
        prev_state: Tensor of shape (B, D), flattened state before transition.
        next_state: Tensor of shape (B, D), flattened state after transition.

    Returns:
        obs_ids: Tensor of shape (B,), each in [0, 2^num_near_peds - 1]
    """
    B = prev_state.shape[0]
    device = prev_state.device
    N = num_near_peds

    # Unflatten states: (B, 1+N, 5)
    prev_state = prev_state.view(B, N + 1, 5)
    next_state = next_state.view(B, N + 1, 5)

    # Extract robot position before transition
    robot_pos_prev = prev_state[:, 0, :2]  # (B, 2)

    # Extract pedestrian states
    ped_prev = prev_state[:, 1:, :]  # (B, N, 5)
    ped_next = next_state[:, 1:, :]  # (B, N, 5)

    # Positions before and after
    ped_pos_prev = ped_prev[:, :, :2]  # (B, N, 2)
    ped_pos_next = ped_next[:, :, :2]  # (B, N, 2)

    # Compute motion vectors
    ped_motion = ped_pos_next - ped_pos_prev  # (B, N, 2)

    # Normalize motion
    motion_dir = ped_motion / (ped_motion.norm(dim=-1, keepdim=True) + 1e-8)

    # Vector from ped to robot (before move)
    rel_to_robot = robot_pos_prev.unsqueeze(1) - ped_pos_prev  # (B, N, 2)
    rel_dir = rel_to_robot / (rel_to_robot.norm(dim=-1, keepdim=True) + 1e-8)

    # Dot product: > 0 → moving toward robot
    dot = (motion_dir * rel_dir).sum(dim=-1)  # (B, N)
    obs_bits = (dot > 0).int()                # (B, N)

    # Convert to integer in [0, 2^N)
    powers = 2 ** torch.arange(N, device=device).unsqueeze(0)  # (1, N)
    obs_ids = (obs_bits * powers).sum(dim=1)  # (B,)

    return obs_ids'''

def observe(prev_state: torch.Tensor, next_state: torch.Tensor, num_near_peds: int, return_obs_bits=False) -> torch.Tensor:
    """
    Computes discrete observation ID for each sample based on whether each of the
    `num_near_peds` pedestrians moved towards or away from the robot.

    Args:
        prev_state: (B, D) flattened state before transition
        next_state: (B, D) flattened state after transition

    Returns:
        obs_ids: (B,), each in [0, 2^num_near_peds - 1]
    """
    B = prev_state.shape[0]
    device = prev_state.device
    N = num_near_peds

    # Unflatten
    prev_state = prev_state.view(B, N + 1, 5)
    next_state = next_state.view(B, N + 1, 5)

    # Robot position (before transition)
    robot_pos = prev_state[:, 0, :2]  # (B, 2)

    # Pedestrian positions
    ped_prev = prev_state[:, 1:, :2]  # (B, N, 2)
    ped_next = next_state[:, 1:, :2]  # (B, N, 2)

    # Distances to robot before and after
    dist_prev = (ped_prev - robot_pos.unsqueeze(1)).norm(dim=-1)  # (B, N)
    dist_next = (ped_next - robot_pos.unsqueeze(1)).norm(dim=-1)  # (B, N)

    # Compare distances
    obs_bits = (dist_next < dist_prev).int()  # (B, N), 1 = moved toward robot

    # Encode binary pattern as int in [0, 2^N)
    powers = 2 ** torch.arange(N, device=device).unsqueeze(0)  # (1, N)
    obs_ids = (obs_bits * powers).sum(dim=1)  # (B,)

    if return_obs_bits:
        return obs_ids, obs_bits, prev_state[:, 1:, 4].long()  # (B, N)
    return obs_ids


def _truncate_to_nearest_peds(state: torch.Tensor, num_peds: int, return_ids: bool = False):
    """
    Truncate a flattened full state (with all peds) to the robot and its `num_peds`
    nearest pedestrians based on Euclidean distance. Pedestrians are returned
    sorted by their ID for consistency.

    Args:
        state: (B, D) flattened state with D = (1 + N) * 5
        num_peds: number of closest pedestrians to retain
        return_ids: if True, also return the ped IDs (float) of the selected peds

    Returns:
        truncated_state: (B, (1 + num_peds) * 5) flattened
        ped_ids (optional): (B, num_peds) tensor of ped IDs (float)
    """
    B, D = state.shape
    assert D % 5 == 0, "State must be flattened with 5 features per agent"
    N = D // 5 - 1  # number of pedestrians

    state = state.view(B, 1 + N, 5)  # (B, 1+N, 5)

    robot_pos = state[:, 0, :2]      # (B, 2)
    ped_pos   = state[:, 1:, :2]     # (B, N, 2)

    dists = torch.norm(ped_pos - robot_pos.unsqueeze(1), dim=-1)  # (B, N)

    # Indices of `num_peds` closest peds
    _, topk_idx = torch.topk(dists, k=num_peds, largest=False)  # (B, num_peds)
    batch_idx = torch.arange(B, device=state.device).unsqueeze(1)  # (B, 1)

    # Select closest peds and their IDs
    closest_peds = state[:, 1:, :][batch_idx, topk_idx]  # (B, num_peds, 5)

    # Sort by pedestrian ID (column 4)
    ped_ids = closest_peds[:, :, 4].long()  # (B, num_peds)
    sorted_ids, sorted_idx = ped_ids.sort(dim=1)
    closest_peds_sorted = closest_peds[batch_idx, sorted_idx]  # (B, num_peds, 5)

    # Combine with robot
    truncated = torch.cat([state[:, :1, :], closest_peds_sorted], dim=1)  # (B, 1 + num_peds, 5)

    if return_ids:
        return truncated.view(B, -1), sorted_ids  # (B, D_trunc), (B, num_peds)
    else:
        return truncated.view(B, -1)  # (B, D_trunc)


class CrowdNav(GenerativeModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gamma = kwargs.get('args_cli').discount_factor
        self._action_ranges = [[i for i in range(5)]]
        self._device = kwargs.get('args_cli').device
        self._headless = kwargs.get('args_cli').headless        
        self._map_size_x = float(kwargs.get('args_cli').map_size_x)
        self._map_size_y = float(kwargs.get('args_cli').map_size_y)
        self._num_peds = int(kwargs.get('args_cli').num_peds)
        self._num_near_peds = int(kwargs.get('args_cli').num_near_peds)
        self._ped_noise_std = 0.05
        #self._ped_noise_std = 0.025        
        self._safety_radius = 1.0
        self._curiosity_distance = float(kwargs.get('args_cli').curiosity_distance)
        self._curiosity_prob = float(kwargs.get('args_cli').curiosity_prob)
        self._move_prob = float(kwargs.get('args_cli').move_prob)
        self._robot_goal = torch.tensor(
            [self._map_size_x, 0.0], 
            dtype=torch.float32, 
            device=self._device,
        )
        self._goal_radius = 2.0

        '''self._step_cost = -1.0
        self._goal_reward = 100.0
        self._collision_penalty = -50.0
        self._yell_penalty = -5.0'''

        self._step_cost = -1.0
        self._goal_reward = 200.0        
        self._collision_penalty = -100.0
        self._yell_penalty = -10.0

        self._step_cost = -1.0
        self._goal_reward = 1000.0        
        self._collision_penalty = -200.0
        self._yell_penalty = -25.0

        if self.role == 'planning':
            self.exec_env = kwargs.get('exec_env')

    @property
    def action_ranges(self):
        return self._action_ranges

    @property
    def state_shape(self):
        return (5*(self._num_near_peds+1),)

    def state_repr(self, state):
        return "None"

    def reset(self):
        """
        Perform reset operations if needed
        """
        num_samples = 1  # always 1 for environment
        device = self._device

        # --- Robot ---
        robot_pos = torch.tensor([1.0, 0.0], dtype=torch.float32, device=device).view(1, 1, 2)
        robot_pos = robot_pos.expand(num_samples, 1, 2)

        robot_speed = torch.zeros((num_samples, 1, 1), dtype=torch.float32, device=device)
        robot_curiosity = -torch.ones_like(robot_speed)  # dummy
        robot_id = -torch.ones((num_samples, 1, 1), dtype=torch.float32, device=device)  # ID = -1

        robot_state = torch.cat([robot_pos, robot_speed, robot_curiosity, robot_id], dim=2)  # (B, 1, 5)

        # --- Pedestrians ---        
        size_x = self._map_size_x 
        x_min = 0.0       
        N = self._num_peds

        positions = torch.empty(num_samples, N, 2, device=device)
        positions[..., 0] = x_min + (size_x - x_min) * torch.rand(num_samples, N, device=device)
        positions[..., 1] = (torch.rand(num_samples, N, device=device) - 0.5) * self._map_size_y

        #speeds = torch.empty((num_samples, N, 1), device=device).uniform_(SPEED_MIN, SPEED_MAX)
        #curiosity = torch.randint(0, 2, (num_samples, N, 1), dtype=torch.float32, device=device)
        # Conditional curiosity sampling
        '''y_coords = positions[..., 1]                          # (1, N)
        p = torch.rand_like(y_coords)                         # (1, N)
        p_correct_curiosity = 1.0

        # Curious if on left and p < 0.8, OR on right and p ≥ 0.8
        is_left = (y_coords >= 0).float()                     # (1, N), 1 if y ≥ 0
        curious_left = (p < p_correct_curiosity).float()                      # 80% curious
        curious_right = (p >= p_correct_curiosity).float()                    # 20% curious
        curiosity_val = is_left * curious_left + (1 - is_left) * curious_right
        curiosity = curiosity_val.unsqueeze(-1) # (1, N, 1)'''
        p_curious = self._curiosity_prob
        curiosity = (torch.rand(num_samples, N, 1, device=device) < p_curious).float()
        speeds = curiosity * CURIOUS_SPEED + (1.0 - curiosity) * SHY_SPEED  # (1, N, 1)

        ped_ids = torch.arange(N, dtype=torch.float32, device=device).view(1, N, 1).expand(num_samples, N, 1)

        ped_state = torch.cat([positions, speeds, curiosity, ped_ids], dim=2)  # (B, N, 5)

        # --- Final state ---
        full_state = torch.cat([robot_state, ped_state], dim=1)  # (B, N+1, 5)
        state = full_state.view(num_samples, -1)  # (B, D)

        self.current_state = state 

        if self.role == "exec":
            self._num_yells = 0
            self._num_collisions = 0

        return True 

    def get_current_state(self):
        return self.current_state 

    def get_previous_state(self):
        return self.previous_state      

    def sample_initial_belief(self, num_samples: int = 1) -> torch.Tensor:
        device = self._device
        K = self._num_near_peds  # number of nearby peds to include

        if self.role == 'planning':
            # 1. Access true pedestrian state from execution environment (flattened)
            exec_state = self.exec_env.get_current_state()  # (1, (1 + N) * 5)
            assert exec_state.ndim == 2 and exec_state.shape[1] == (1 + self._num_peds) * 5

            # 2. Truncate to robot + nearest K pedestrians (returned as flattened)
            trunc_flat = _truncate_to_nearest_peds(exec_state, K)  # (1, (1 + K) * 5)

            # 3. Unpack flattened truncated state
            trunc = trunc_flat.view(1, 1 + K, 5)
            robot = trunc[:, 0, :]              # (1, 5)
            peds = trunc[:, 1:, :]              # (1, K, 5)

            robot_pos = robot[:, :2]            # (1, 2)
            robot_id = robot[:, 4:5]            # (1, 1)

            ped_pos = peds[:, :, :2]            # (1, K, 2)
            ped_speed = peds[:, :, 2:3]         # (1, K, 1)
            ped_ids = peds[:, :, 4:5]           # (1, K, 1)

            # 4. Sample curiosity for each pedestrian
            curiosity = torch.randint(
                0, 2, size=(num_samples, K, 1),
                device=device, dtype=torch.float32
            )  # (B, K, 1)

            # 5. Expand static info to batch
            ped_pos = ped_pos.expand(num_samples, -1, -1)     # (B, K, 2)
            ped_speed = ped_speed.expand(num_samples, -1, -1) # (B, K, 1)
            ped_ids = ped_ids.expand(num_samples, -1, -1)     # (B, K, 1)

            ped_state = torch.cat([ped_pos, ped_speed, curiosity, ped_ids], dim=2)  # (B, K, 5)

            # 6. Robot state: [x, y, speed=0, curiosity=-1, id=-1]
            robot_pos = robot_pos.expand(num_samples, -1)         # (B, 2)
            robot_speed = torch.zeros((num_samples, 1), device=device)
            robot_curiosity = -torch.ones((num_samples, 1), device=device)
            robot_id = robot_id.expand(num_samples, -1)           # (B, 1)
            robot_state = torch.cat([robot_pos, robot_speed, robot_curiosity, robot_id], dim=1).unsqueeze(1)  # (B, 1, 5)

            # 7. Concatenate and flatten
            full_state = torch.cat([robot_state, ped_state], dim=1)  # (B, 1+K, 5)
            state = full_state.view(num_samples, -1)                 # (B, D)
            return state

        else:
            # Exec environment: reuse true full current state (already flattened)
            return self.current_state

    def get_ped_ids_obs_exec(self):
        return self.ped_ids_obs_exec

    def get_obs_bits(self):
        return self.obs_bits, self.obs_ids

    def sample(self, state: torch.Tensor, action: torch.Tensor, **kwargs) -> dict:
        """
        Simulates environment transition given a flattened state and action.

        Args:
            state: (B, D) flattened state [robot, peds]
            action: (B,) long tensor of discrete movement directions (0=N, 1=E, 2=S, 3=W)

        Returns:
            Dict with next_state, observation, reward, terminal, nsteps, info
        """
        B, D = state.shape
        device = state.device

        # Step pedestrians: FLATTENED in, FLATTENED out
        next_state = step_pedestrians(
            state,
            action, 
            self._map_size_x, 
            self._map_size_y, 
            noise_std=self._ped_noise_std,
            move_prob=self._move_prob,
            curiosity_distance=self._curiosity_distance,
        )  # (B, D)

        # Temporarily unflatten for robot manipulation
        state_unflat = state.view(B, -1, 5)       # (B, 1+N, 5)
        next_unflat = next_state.view(B, -1, 5)   # (B, 1+N, 5)

        # Move robot
        robot_pos = state_unflat[:, 0, :2]  # (B, 2)
        MOVE_DIRS = torch.tensor([
            [1.0,  0.0],   # North
            [0.0, -1.0],   # East
            [-1.0, 0.0],   # South
            [0.0,  1.0],   # West,
            [0.0,  0.0],   # Yell = no movement
        ], device=device, dtype=robot_pos.dtype)  # (4, 2)

        displacement = MOVE_DIRS[action.squeeze(-1).long()]  # (B, 2)
        new_robot_pos = robot_pos + displacement             # (B, 2)

        # Clamp to map bounds
        half_y = self._map_size_y / 2.0
        new_robot_pos[..., 0] = torch.clamp(new_robot_pos[..., 0], 0.0, self._map_size_x)       # x-axis
        new_robot_pos[..., 1] = torch.clamp(new_robot_pos[..., 1], -half_y, +half_y)            # y-axis

        # Update robot in next_state
        next_unflat[:, 0, :2] = new_robot_pos               # Update position
        next_unflat[:, 0, 2:] = state_unflat[:, 0, 2:]      # Retain speed, curiosity, ID

        # Re-flatten
        next_state = next_unflat.view(B, -1)

        # Truncate both state and next_state if in exec role
        if self.role == 'exec':
            next_state_obs, nearest_ids = _truncate_to_nearest_peds(
                next_state, 
                self._num_near_peds, 
                return_ids=True
            )  # (B, (1+K)*5), (B, K)
            state_full = state.view(B, -1, 5)  # (B, 1+N, 5)

            # Get all ped IDs from state (excluding robot)
            state_ped_ids = state_full[:, 1:, 4]  # (B, N)
            ped_idx = (nearest_ids.unsqueeze(2) == state_ped_ids.unsqueeze(1)).nonzero(as_tuple=False)  # (B*K, 3)
            ped_idx = ped_idx[:, 2].view(B, -1)  # (B, K)
            batch_idx = torch.arange(B, device=state.device).unsqueeze(1)  # (B, 1)
            matched_peds = state_full[:, 1:, :][batch_idx, ped_idx]  # (B, K, 5)

            state_obs = torch.cat([state_full[:, :1, :], matched_peds], dim=1)  # (B, 1+K, 5)
            state_obs = state_obs.view(B, -1)  # (B, (1+K)*5)

            # In sample()
            N = self._num_near_peds
            state_obs_unflat = state_obs.view(B, N + 1, 5)
            ped_ids_obs = state_obs_unflat[:, 1:, 4].long()

            print("[DEBUG] Observation generated from these pedestrian IDs (first sample):")
            print(ped_ids_obs[0].tolist())  # (N,)

            self.ped_ids_obs_exec = ped_ids_obs[0].tolist()

            '''state_obs = _truncate_to_nearest_peds(state, self._num_near_peds)        # (B, (1+K)*5)
            next_state_obs = _truncate_to_nearest_peds(next_state, self._num_near_peds)'''            
        else:                            
            state_obs = state
            next_state_obs = next_state
            # === NEW DEBUG BLOCK ===
            if kwargs.get('belief_update', False):
                N = self._num_near_peds
                state_obs_unflat = state_obs.view(B, N + 1, 5)
                next_state_obs_unflat = next_state_obs.view(B, N + 1, 5)

                state_ids = state_obs_unflat[:, 1:, 4].long()     # (B, N)
                next_ids = next_state_obs_unflat[:, 1:, 4].long() # (B, N)

                print("[DEBUG] Belief update mode: Using these pedestrian IDs from state_obs (first sample):")
                print(state_ids[0].tolist())

                print("[DEBUG] Belief update mode: Using these pedestrian IDs from next_state_obs (first sample):")
                print(next_ids[0].tolist())

        # Observation (uses flattened state)
        obs, obs_bits, obs_ids = observe(state_obs, next_state_obs, self._num_near_peds, return_obs_bits=True)  # (B,)
        if self.role == 'exec':
            self.obs_bits = obs_bits
            self.obs_ids = obs_ids

        # Reward computation
        next_unflat = next_state.view(B, -1, 5)
        robot_pos = next_unflat[:, 0, :2]      # (B, 2)
        ped_pos   = next_unflat[:, 1:, :2]     # (B, N, 2)

        dists = torch.norm(ped_pos - robot_pos.unsqueeze(1), dim=-1)  # (B, N)
        min_dist, _ = dists.min(dim=1)
        collision = (min_dist <= self._safety_radius)

        #goal_dists = torch.norm(robot_pos - self._robot_goal.unsqueeze(0), dim=-1)  # (B,)
        #reached_goal = (goal_dists <= self._goal_radius)
        reached_goal = (robot_pos[:, 0] > self._map_size_x - 0.01)

        reward = self._step_cost * torch.ones(B, device=device)
        reward[collision] = self._collision_penalty
        reward[reached_goal] += self._goal_reward

        # === YELL PENALTY ===
        YELL_IDX = 4  # Assumes YELL is the 5th discrete action
        is_yell = (action.squeeze(-1) == YELL_IDX)  # (B,)
        reward[is_yell] += self._yell_penalty  # Apply penalty (should be negative)

        # Track state in execution role
        if self.role == 'exec':
            self.previous_state = self.current_state.clone()
            self.current_state = next_state.clone()            
            if is_yell.cpu().item():
                self._num_yells  = self._num_yells + 1
            if collision.cpu().item():
                self._num_collisions = self._num_collisions + 1                    

        return {
            "next_state": next_state,         # (B, D)
            "observation": obs,               # (B,)
            "reward": reward,                 # (B,)
            "terminal": reached_goal.clone(), # (B,)
            "nsteps": 1,
            "info": {}
        }


    def heuristics(self, state: torch.Tensor, action: torch.Tensor, current_nodes: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Heuristic upper bound on expected return: assumes robot moves straight to goal,
        ignoring all pedestrians (i.e., best-case optimistic estimate).

        Args:
            state: (B, D) flattened state
            action: (B,) integer actions (not used here)
            current_nodes: Optional (not used here)

        Returns:
            Tensor of shape (B,) with heuristic value estimates
        """
        B = state.shape[0]
        device = state.device

        # Extract robot position (x, y) from first 2 elements
        robot_pos = state[:, 0:2]  # (B, 2)

        # L1 distance to goal
        #d1 = (robot_pos - self._robot_goal).abs().sum(dim=1)  # (B,)
        d1 = (state[:, 0] - self._robot_goal[0]).abs()               # (B,)
        steps_lb = torch.clamp(d1 - self._goal_radius, min=0.0)
        T = torch.ceil(steps_lb)  # optimistic minimum steps to reach goal

        gamma = torch.tensor(self.gamma, device=device, dtype=torch.float32)
        gamma_T = torch.pow(gamma, T)

        step_sum = self._step_cost * (1.0 - gamma_T) / (1.0 - gamma)
        total = step_sum + self._goal_reward * gamma_T
        return total  # (B,)

    '''def likelihood(
        self, 
        observation: torch.Tensor,
        prev_state: torch.Tensor, 
        next_state: torch.Tensor, 
        action: torch.Tensor, 
        log_likelihood: bool = False,
        is_encoded_observation: bool = True,  # ignored
    ) -> torch.Tensor:
        """
        Compute p(o_env | s, a, s') under a noisy binary observation model.
        Returns high probability if the nominal observation from (s, a, s') matches
        the true observation, otherwise low probability (to avoid particle impoverishment).
        """
        B = next_state.shape[0]
        device = next_state.device
        N = self._num_near_peds
        p_correct = 0.95  # Soft match likelihood

        # Expand observation if needed
        if observation.ndim == 0:
            observation = observation.unsqueeze(0).expand(B)
        elif observation.shape[0] == 1 and B > 1:
            observation = observation.expand(B)
        else:
            assert observation.shape[0] == B, \
                f"Observation batch size {observation.shape[0]} does not match B={B}"

        # Compute nominal observation under this particle
        nominal_obs = observe(prev_state, next_state, N)  # (B,)

        # Check for match
        match = (nominal_obs == observation)  # (B,)
        result = torch.where(
            match, 
            torch.tensor(p_correct, device=device),
            torch.tensor(1.0 - p_correct, device=device),
        )

        if log_likelihood:
            result = torch.log(result + 1e-18)

        return result'''

    def likelihood(
        self,
        observation: torch.Tensor,
        prev_state: torch.Tensor,
        next_state: torch.Tensor,
        action: torch.Tensor,
        log_likelihood: bool = False,
        is_encoded_observation: bool = True,  # Ignored
    ) -> torch.Tensor:
        """
        Compute p(o | s, a, s') under a noisy bitwise binary observation model.
        Each ped observation bit contributes independently with probability p_correct.
        """
        B = next_state.shape[0]
        device = next_state.device
        N = self._num_near_peds
        p_correct = 0.8

        # Expand observation if needed
        if observation.ndim == 0:
            observation = observation.unsqueeze(0).expand(B)
        elif observation.shape[0] == 1 and B > 1:
            observation = observation.expand(B)
        else:
            assert observation.shape[0] == B, \
                f"Observation batch size {observation.shape[0]} does not match B={B}"

        obs_bits_true, obs_ids_true = self.exec_env.get_obs_bits()  # (B, N_true)

        # Predicted observation bits and pedestrian IDs from simulated state
        _, obs_bits_pred, pred_ids = observe(prev_state, next_state, N, return_obs_bits=True)  # (B, N_pred)

        # Match each predicted ID against true IDs
        match = pred_ids.unsqueeze(2) == obs_ids_true.unsqueeze(1)  # (B, N_pred, N_true)
        any_match = match.any(dim=2)                                # (B, N_pred)
        matched_idx = match.float().argmax(dim=-1)                  # (B, N_pred)

        # Gather true bits for matched IDs (safe since unmatched will be ignored later)
        obs_bits_true = obs_bits_true.expand(B, -1)
        matched_true_bits = obs_bits_true.gather(dim=1, index=matched_idx)

        # Compare predicted vs matched true bits
        bit_matches = (obs_bits_pred == matched_true_bits).float()  # (B, N_pred)

        # Set p = 1 for matching bits, 0 otherwise (you can adjust p_correct if needed)        
        bitwise_probs = bit_matches * p_correct + (1 - bit_matches) * (1 - p_correct)

        # Mask: set likelihood to 1.0 for unmatched peds (i.e., peds not in true obs)
        bitwise_probs = bitwise_probs * any_match.float() + (~any_match).float()

        # Final likelihood
        if log_likelihood:
            bitwise_probs = torch.clamp(bitwise_probs, min=1e-8)  # avoid log(0)
            result = bitwise_probs.log().sum(dim=1)  # (B,)
        else:
            result = bitwise_probs.prod(dim=1)  # (B,)


        self.print_avg_curiosities(prev_state, "likelihood prev_state")

        # === DEBUG: Compare ped IDs in prev_state and next_state ===
        with torch.no_grad():
            D_ped = 5
            B, D = prev_state.shape
            N = self._num_near_peds

            prev_ids = prev_state.view(B, 1 + N, D_ped)[:, 1:, 4].long()  # (B, N)
            next_ids = next_state.view(B, 1 + N, D_ped)[:, 1:, 4].long()  # (B, N)

            # Check first particle only (or more if you want)
            print("\n[DEBUG] Compare ped IDs in first particle:")
            print(f"  prev_state IDs: {prev_ids[0].tolist()}")
            print(f"  next_state IDs: {next_ids[0].tolist()}")
            print()

            # Optional: check if they are identical across all B
            id_match = (prev_ids == next_ids).all(dim=1)  # (B,)
            num_mismatched = (~id_match).sum().item()
            if num_mismatched > 0:
                print(f"[DEBUG] ⚠️  {num_mismatched}/{B} particles have mismatched ped IDs between prev_state and next_state.")
            else:
                print("[DEBUG] ✅ All particles have matching ped IDs.")

        with torch.no_grad():
            B, D = next_state.shape
            D_ped = 5
            N = self._num_near_peds

            prev_ids = prev_state.view(B, N + 1, D_ped)[:, 1:, 4].long()  # (B, N)
            next_ids = next_state.view(B, N + 1, D_ped)[:, 1:, 4].long()  # (B, N)

            same_order = (prev_ids == next_ids).all(dim=1)  # (B,) → True if ped order matches for each sample
            all_match = same_order.all().item()

            print("\n[DEBUG] Check pedestrian ordering between prev_state and next_state:")
            if all_match:
                print("  ✅ All particles have matching ped ID ordering.")
            else:
                mismatch_count = (~same_order).sum().item()
                print(f"  ❌ {mismatch_count} / {B} particles have mismatched ped ID order.")
                # Optionally show example mismatch
                idx = (~same_order).nonzero(as_tuple=False)[0].item()
                print(f"  Example mismatch at particle {idx}:")
                print(f"    prev_ids = {prev_ids[idx].tolist()}")
                print(f"    next_ids = {next_ids[idx].tolist()}")

        # === Weighted average curiosity per ped ID across all particles ===
        with torch.no_grad():
            B, D = next_state.shape
            D_ped = 5
            N = self._num_near_peds
            next_state_unflat = next_state.view(B, 1 + N, D_ped)

            ped_curiosity = next_state_unflat[:, 1:, 3]  # (B, N)
            ped_ids = next_state_unflat[:, 1:, 4].long()  # (B, N)

            weights = result.exp()  # (B,)
            weights = weights / (weights.sum() + 1e-8)  # Normalize

            # Flatten
            flat_ids = ped_ids.reshape(-1)              # (B*N,)
            flat_curiosity = ped_curiosity.reshape(-1)  # (B*N,)
            flat_weights = weights.repeat_interleave(N)  # (B*N,)

            # Unique IDs
            unique_ids, inverse = torch.unique(flat_ids, return_inverse=True)
            num_ids = unique_ids.shape[0]

            # Weighted sum and count
            weighted_sums = torch.zeros(num_ids, device=next_state.device).scatter_add(0, inverse, flat_curiosity * flat_weights)
            weight_totals = torch.zeros(num_ids, device=next_state.device).scatter_add(0, inverse, flat_weights)
            avg_curiosity = weighted_sums / (weight_totals + 1e-8)

            # Sort by ID
            sorted_vals = sorted(zip(unique_ids.tolist(), avg_curiosity.tolist()), key=lambda x: x[0])

            # Pretty print
            print(f"[DEBUG] weighted curiosity per ped ID across all particles:")
            for pid, avg_c in sorted_vals:
                print(f"  ID={pid:<3} → avg_curiosity={avg_c:.2f}")


        return result

    def print_avg_curiosities(self, belief_particles: torch.Tensor, prefix: str = ""):
        """
        Print average curiosity values per unique ped ID across all belief particles.
        Assumes belief_particles shape is (B, D) where D = (1 + N) * 5.
        """
        B, flat_D = belief_particles.shape
        D = 5
        N = self._num_near_peds

        belief_reshaped = belief_particles.view(B, 1 + N, D)
        belief_ids = belief_reshaped[:, 1:, 4].long().flatten()     # (B * N,)
        belief_c = belief_reshaped[:, 1:, 3].flatten()              # (B * N,)

        unique_ids, inverse = torch.unique(belief_ids, return_inverse=True)
        sum_c = torch.zeros_like(unique_ids, dtype=torch.float32)
        count = torch.zeros_like(unique_ids, dtype=torch.float32)

        sum_c = sum_c.index_add(0, inverse, belief_c)
        count = count.index_add(0, inverse, torch.ones_like(belief_c))
        avg_c_per_id = sum_c / count

        avg_curiosity_dict = {
            int(pid.item()): float(c.item())
            for pid, c in zip(unique_ids, avg_c_per_id)
        }

        print(f"\n[DEBUG] {prefix} Avg curiosity per ped ID across all particles:")
        for pid, c in avg_curiosity_dict.items():
            print(f"  ID={pid:2d} → avg_curiosity={c:.2f}")
        print("------------------------------------------------------\n")

        return avg_curiosity_dict  # optional: useful if you want to inspect in code

    def postprocess_belief_particles(
        self,
        prior_belief_particles: torch.Tensor,
        action: torch.Tensor,
        observation: torch.Tensor,
        belief_particles: torch.Tensor,
    ) -> torch.Tensor:
        """
        Updates belief particles by:
        - Keeping robot unchanged
        - Replacing pedestrian subset with closest ones from current true state
        - Reusing curiosity for matching IDs and sampling new values for others
        """
        self.print_avg_curiosities(belief_particles, "postprocess_belief_particles(belief_particles)")

        B, flat_D = prior_belief_particles.shape
        D = 5
        N = self._num_near_peds
        device = prior_belief_particles.device

        # 1. Get true states
        current_state = self.exec_env.get_current_state().expand(B, -1)
        curr, nearest_ids = _truncate_to_nearest_peds(current_state, N, return_ids=True)
        curr = curr.view(B, 1 + N, D)

        # 2. Reuse curiosity from current belief if ID matches
        prior = belief_particles.view(B, 1 + N, D)
        prior_ids = prior[:, 1:, 4].long()       # (B, N)
        prior_c = prior[:, 1:, 3]                # (B, N)

        match_prior = (nearest_ids.unsqueeze(2) == prior_ids.unsqueeze(1))  # (B, N, N)
        reuse_prior_any = match_prior.any(dim=2)
        reuse_prior_idx = match_prior.float().argmax(dim=-1)

        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, N)
        curiosity_reused = torch.zeros(B, N, device=device)
        curiosity_reused[reuse_prior_any] = prior_c[batch_idx[reuse_prior_any], reuse_prior_idx[reuse_prior_any]]

        # 3. Sample random curiosity {0,1} for new IDs
        rand_curiosity = torch.randint(0, 2, (B, N), device=device, dtype=torch.float)
        curiosity_final = torch.where(reuse_prior_any, curiosity_reused, rand_curiosity)

        # 4. Construct new belief state: keep robot, replace pedestrians
        robot = curr[:, :1, :]          # (B, 1, D)
        peds = curr[:, 1:, :].clone()   # (B, N, D)
        peds[:, :, 3] = curiosity_final
        updated_belief = torch.cat([robot, peds], dim=1)  # (B, 1+N, D)

        # === Average curiosity per ped ID ===
        all_ids = updated_belief[:, 1:, 4].long().view(-1)         # (B*N,)
        all_curiosity = updated_belief[:, 1:, 3].reshape(-1)          # (B*N,)

        unique_ids, inverse_idx = torch.unique(all_ids, return_inverse=True)
        return updated_belief.view(B, -1)

    def debug_belief(self, state: torch.Tensor, observation: torch.Tensor, belief: torch.Tensor):
        B = state.shape[0]
        N = self._num_near_peds

        with torch.no_grad():
            state_tensor = state.view(B, -1, 5)
            true_curiosities = state_tensor[0, 1:, 3]
            true_ids = state_tensor[0, 1:, 4].long()

            belief_reshaped = belief.view(-1, 1 + N, 5)
            all_ids = belief_reshaped[:, 1:, 4].long().flatten()
            all_c = belief_reshaped[:, 1:, 3].flatten()
            unique_ids, inverse = torch.unique(all_ids, return_inverse=True)
            sum_c = torch.zeros_like(unique_ids, dtype=torch.float32)
            count = torch.zeros_like(unique_ids, dtype=torch.float32)
            sum_c = sum_c.index_add(0, inverse, all_c)
            count = count.index_add(0, inverse, torch.ones_like(all_c))
            avg_c_per_id = sum_c / count
            avg_curiosity_dict = {int(pid.item()): float(c.item()) for pid, c in zip(unique_ids, avg_c_per_id)}

            belief_tensor = belief.view(-1, 1 + N, 5)[0]
            belief_ids = belief_tensor[1:, 4].long().cpu()
            avg_curiosity = torch.tensor([
                avg_curiosity_dict.get(int(pid), 0.0) for pid in belief_ids
            ])

            '''print(f"\n[DEBUG] First particle belief IDs: {belief_ids.tolist()}")
            print(f"[DEBUG] Unique IDs in all particles: {unique_ids.tolist()}")
            print(f"[DEBUG] Avg curiosity per unique ID:")
            for pid, c in avg_curiosity_dict.items():
                print(f"  ID={pid:2d} → avg_curiosity={c:.2f}")

            print(f"[DEBUG] Curiosity values assigned to first particle's IDs: {avg_curiosity.tolist()}")
            print(f"[DEBUG] Curiosity values assigned to first particle's IDs: {[avg_curiosity_dict[i.item()] for i in belief_ids]}")'''

            # --- Decode observation bits ---
            obs_id = observation[0].item()  # Assuming B=1
            obs_bits = [(obs_id >> i) & 1 for i in range(N)]  # List of bits, LSB = ped 0

            print("--- Observation Bits (movement relative to robot) ---")
            for i, bit in enumerate(obs_bits):
                direction = "closer" if bit == 1 else "away"
                print(f"Belief Ped {i:2d} (ID={belief_ids[i].item():2d}): moved {direction}")
            print("------------------------------------------------------\n")


            print("\n--- Curiosity Debug (matched by ID) ---")
            # --- Collect all debug info ---
            debug_info = []
            for i in range(N):
                pid = belief_ids[i].item()
                match_idx = (true_ids == pid).nonzero(as_tuple=False)
                true_c = float("nan") if match_idx.numel() == 0 else true_curiosities[match_idx[0, 0]].item()
                inferred_c = avg_curiosity[i].item()
                label = "curious" if inferred_c > 0.5 else "non-curious"
                debug_info.append((pid, true_c, inferred_c, label))

            # --- Sort by ped ID ---
            debug_info.sort(key=lambda x: x[0])

            # --- Print nicely ---
            for pid, true_c, inferred_c, label in debug_info:
                print(f"Belief Ped (ID={pid:2d}): true = {true_c:.2f}, "
                      f"inferred = {inferred_c:.2f}, label = {label}")

            print("--------------------------------------------------\n")
        curious_mask = (avg_curiosity > 0.5)
        return avg_curiosity_dict, avg_curiosity, curious_mask

    '''def is_goal(self, state: torch.Tensor) -> torch.Tensor:
        B = state.shape[0]
        state_unflat = state.view(B, -1, 5)
        robot_pos = state_unflat[:, 0, :2]
        goal_dists = torch.norm(robot_pos - self._robot_goal.unsqueeze(0), dim=-1)  # (B,)
        reached_goal = (goal_dists <= self._goal_radius)
        return reached_goal'''
    def is_goal(self, state: torch.Tensor) -> torch.Tensor:
        B = state.shape[0]
        state_unflat = state.view(B, -1, 5)
        robot_pos = state_unflat[:, 0, :2]  # (B, 2)
        
        # Check if x-coordinate exceeds map width threshold
        reached_goal = (robot_pos[:, 0] > self._map_size_x - 0.01)  # (B,)
        
        return reached_goal


    def plot(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor, 
        observation: torch.Tensor, 
        belief: torch.Tensor
    ):
        if self._headless:
            return

        B, D = state.shape
        N = self._num_near_peds
        n_plotted_particles = 5

        s_np = state.view(B, -1, 5)[0].detach().cpu().numpy()
        robot_pos = s_np[0, :2]
        ped_pos = s_np[1:, :2]

        if not hasattr(self, "_fig") or self._fig is None:
            plt.ion()
            self._fig, self._ax = plt.subplots(1, 1, figsize=(6, 6))
            ax = self._ax
            ax.set_aspect('equal')
            half_y = self._map_size_y / 2.0
            ax.set_xlim(half_y, -half_y)
            ax.set_ylim(0, self._map_size_x) 
            ax.set_xlabel("y")
            ax.set_ylabel("x")
            ax.set_title("Environment State")

            gx, gy = self._robot_goal.cpu().numpy()
            self._goal_circle = Circle((gy, gx), radius=self._goal_radius, color='green', alpha=0.3)
            ax.add_patch(self._goal_circle)

            self._robot_safe = Circle((robot_pos[1], robot_pos[0]), radius=self._safety_radius, color='black', alpha=0.9)
            ax.add_patch(self._robot_safe)

            self._ped_circles = [Circle((0, 0), radius=0.3, color='red', alpha=0.8, visible=False) for _ in range(len(ped_pos))]
            for c in self._ped_circles:
                ax.add_patch(c)

            self._belief_circles = [
                Circle((0, 0), radius=0.6, color='orange', alpha=0.7, linewidth=0, visible=False)
                for _ in range(n_plotted_particles * N)
            ]

            self._belief_labels = [
                ax.text(0, 0, "", fontsize=6, ha="center", va="bottom", visible=False)
                for _ in range(n_plotted_particles * N)
            ]

            for c in self._belief_circles:
                ax.add_patch(c)

            self._action_arrow = ax.arrow(0, 0, 0, 0, head_width=0.5, head_length=0.5, fc='blue', ec='blue', visible=False)

            self._yell_label = ax.text(
                robot_pos[1], robot_pos[0] + 0.5,  # (y, x + offset)
                "Back off!",
                fontsize=10,
                color="purple",
                ha="center",
                va="bottom",
                visible=False,
                fontweight='bold'
            )


        ax = self._ax
        self._robot_safe.center = (robot_pos[1], robot_pos[0])

        for i, c in enumerate(self._ped_circles):
            if i < len(ped_pos):
                c.center = (ped_pos[i, 1], ped_pos[i, 0])
                c.set_visible(True)
            else:
                c.set_visible(False)

        if belief is not None and belief.numel() > 0:
            avg_curiosity_dict, avg_curiosity, curious_mask = self.debug_belief(state, observation, belief)

            belief_reshaped = belief.view(-1, 1 + N, 5)
            b_pos_np = belief_reshaped[:n_plotted_particles, 1:, :2].cpu().numpy()
            b_ids_np = belief_reshaped[:n_plotted_particles, 1:, 4].cpu().numpy().astype(int)  # shape: (P, N)
            flat_ids = b_ids_np.reshape(-1)  # shape: (P * N,)
            flat_belief = b_pos_np.reshape(-1, 2)

            for i, c in enumerate(self._belief_circles):
                if i < len(flat_belief):
                    c.center = (flat_belief[i, 1], flat_belief[i, 0])
                    ped_idx = i % N
                    c.set_color("blue" if curious_mask[ped_idx] else "green")
                    c.set_visible(True)
                    # Update corresponding label
                    label = self._belief_labels[i]
                    label.set_position((flat_belief[i, 1], flat_belief[i, 0]))
                    label.set_text(str(flat_ids[i]))
                    label.set_visible(True)
                else:
                    c.set_visible(False)
                    self._belief_labels[i].set_visible(False)
        else:
            for c, t in zip(self._belief_circles, self._belief_labels):
                c.set_visible(False)
                t.set_visible(False)
            '''for c in self._belief_circles:
                c.set_visible(False)'''

        if action is not None:
            is_yell = (action.squeeze().item() == 4)  # assuming index 4 is YELL
            self._yell_label.set_position((robot_pos[1], robot_pos[0] + 0.5))  # position above robot
            self._yell_label.set_visible(is_yell)
        else:
            self._yell_label.set_visible(False)

        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        #input('go on')

    def get_info(self) -> str:
        if self.role == "exec":
            return f"num yells = {self._num_yells}, num_collision = {self._num_collisions}"
        return ""