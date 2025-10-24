import time
import torch

class Tree:
    def __init__(self, device=None, init_capacity: int = 1024):
        # Initialize the tree with just the root node
        self.device = device
        self.data_shape = 2 # Immediate rewards & visit counts

        self.action_capacity = init_capacity
        self.num_action_nodes = 0

        self.obs_capacity = init_capacity
        self.num_obs_nodes = 1  # root node will be inserted immediately

        # Preallocate action_nodes
        self._action_nodes = torch.empty(
            (self.action_capacity, 2 + self.data_shape),  # [parent, value, reward, visits]
            dtype=torch.float32,
            device=device
        )

        # Preallocate observation_nodes
        self._observation_nodes = torch.empty(
            (self.obs_capacity, 2 + self.data_shape),     # [parent, value, reward, visits]
            dtype=torch.float32,
            device=device
        )

        # Insert root observation node at index 0
        self._observation_nodes[0, :2] = torch.tensor([-1, -1], dtype=torch.float32, device=device)
        self._observation_nodes[0, 2:] = 0.0

    @property
    def action_nodes(self):
        return self._action_nodes[:self.num_action_nodes]

    @property
    def observation_nodes(self):
        return self._observation_nodes[:self.num_obs_nodes]    

    def update_current_node_visit_count(self, current_nodes):
        current_nodes_unique, inverse = torch.unique(current_nodes, return_inverse=True)
        visit_counts = torch.zeros_like(current_nodes_unique, dtype=torch.float32)
        visit_counts.scatter_add_(0, inverse, torch.ones_like(inverse, dtype=torch.float32))
        self._observation_nodes[current_nodes_unique, 3] += visit_counts

    def insert_action(self, parent_nodes, actions, rewards, terminal):
        unique_pairs, inverse, matched_indices, new_mask = self._dedup_and_match(
            parent_nodes, actions, self._action_nodes, self.num_action_nodes
        )

        # Step 3: insert new pairs if needed
        num_new = new_mask.sum().item()
        if num_new > 0:
            self._ensure_capacity("action", num_new)
            start = self.num_action_nodes
            end = start + num_new
            self._action_nodes[start:end, :2] = unique_pairs[new_mask].to(torch.float32)
            self._action_nodes[start:end, 2:] = 0.0
            new_insert_indices = torch.arange(start, end, device=self.device)
            self.num_action_nodes = end
        else:
            new_insert_indices = torch.empty(0, dtype=torch.long, device=self.device)

        # Step 4: map batch back to indices
        child_indices_unique = torch.empty(unique_pairs.shape[0], dtype=torch.long, device=self.device)
        if matched_indices.numel() > 0:
            child_indices_unique[~new_mask] = matched_indices
        if new_insert_indices.numel() > 0:
            child_indices_unique[new_mask] = new_insert_indices
        child_indices = child_indices_unique[inverse]

        # Step 5: update rewards + visits
        unique_child, inv_idx = torch.unique(child_indices, return_inverse=True)
        self._action_nodes[unique_child, 2] += torch.bincount(inv_idx, weights=rewards)
        visit_counts = torch.bincount(inv_idx)
        self._action_nodes[unique_child, 3] += visit_counts.to(torch.float32)

        return child_indices[~terminal.view(-1)]

    def insert_observation(self, parent_nodes, observations):
        unique_pairs, inverse, matched_indices, new_mask = self._dedup_and_match(
            parent_nodes, observations, self._observation_nodes, self.num_obs_nodes
        )

        # Step 3: insert new pairs if needed
        num_new = new_mask.sum().item()
        if num_new > 0:
            self._ensure_capacity("obs", num_new)
            start = self.num_obs_nodes
            end = start + num_new
            self._observation_nodes[start:end, :2] = unique_pairs[new_mask].to(torch.float32)
            self._observation_nodes[start:end, 2:] = 0.0
            new_insert_indices = torch.arange(start, end, device=self.device)
            self.num_obs_nodes = end
        else:
            new_insert_indices = torch.empty(0, dtype=torch.long, device=self.device)

        # Step 4: map batch back to indices
        child_indices_unique = torch.empty(unique_pairs.shape[0], dtype=torch.long, device=self.device)
        if matched_indices.numel() > 0:
            child_indices_unique[~new_mask] = matched_indices
        if new_insert_indices.numel() > 0:
            child_indices_unique[new_mask] = new_insert_indices
        obs_indices = child_indices_unique[inverse]

        return obs_indices

    def update_leaf_values(self, heuristic, current_nodes):
        unique_nodes, inverse = torch.unique(current_nodes, return_inverse=True)        
        leaf_values = torch.zeros_like(unique_nodes, dtype=torch.float32)
        leaf_values.scatter_add_(0, inverse, heuristic)
        self._observation_nodes[unique_nodes, 2] += leaf_values

        visit_counts = torch.zeros_like(unique_nodes, dtype=torch.float32)
        visit_counts.scatter_add_(0, inverse, torch.ones_like(inverse, dtype=torch.float32))
        self._observation_nodes[unique_nodes, 3] += visit_counts

    def print(self):
        old_precision = torch._tensor_str.PRINT_OPTS.precision
        old_sci_mode = torch._tensor_str.PRINT_OPTS.sci_mode
        torch.set_printoptions(precision=8, sci_mode=False)        
        for i in range(self.nodes.shape[0]):
            print(f"{i}: {self.nodes[i]}")

        torch.set_printoptions(precision=old_precision, sci_mode=old_sci_mode)

    def _ensure_capacity(self, node_type: str, num_new: int):
        """
        Ensure there is enough buffer capacity for inserting num_new nodes.
        If not, grow to either double the current capacity or the exact size needed,
        whichever is larger.
        """
        if node_type == "action":
            needed = self.num_action_nodes + num_new
            if needed > self.action_capacity:
                new_cap = max(self.action_capacity * 2, needed)
                new_buf = torch.empty(
                    (new_cap, 2 + self.data_shape),
                    dtype=self._action_nodes.dtype,
                    device=self.device,
                )
                new_buf[:self.num_action_nodes] = self._action_nodes[:self.num_action_nodes]
                self._action_nodes = new_buf
                self.action_capacity = new_cap

        elif node_type == "obs":
            needed = self.num_obs_nodes + num_new
            if needed > self.obs_capacity:
                new_cap = max(self.obs_capacity * 2, needed)
                new_buf = torch.empty(
                    (new_cap, 2 + self.data_shape),
                    dtype=self._observation_nodes.dtype,
                    device=self.device,
                )
                new_buf[:self.num_obs_nodes] = self._observation_nodes[:self.num_obs_nodes]
                self._observation_nodes = new_buf
                self.obs_capacity = new_cap

    def _dedup_and_match(self, parent_indices: torch.Tensor, values: torch.Tensor,
                         existing_nodes: torch.Tensor, num_existing: int):
        """
        Deduplicate (parent, value) pairs and match them against existing nodes.

        Args:
            parent_indices: (N,) tensor of parent IDs
            values: (N,) tensor of action or observation IDs
            existing_nodes: (M, 2) tensor of existing (parent, value) pairs
            num_existing: number of valid rows in existing_nodes

        Returns:
            unique_pairs: (K, 2) deduplicated (parent, value) pairs
            inverse: (N,) mapping from input rows -> unique_pairs rows
            matched_indices: (L,) indices in existing_nodes that matched
            new_mask: (K,) bool mask, True where unique_pairs are new
        """
        # Step 1: deduplicate
        parent_and_val = torch.stack((parent_indices, values), dim=1)
        unique_pairs, inverse = torch.unique(parent_and_val, dim=0, return_inverse=True)

        # Step 2: match vs existing
        if num_existing > 0:
            existing_pairs = existing_nodes[:num_existing, :2].to(torch.int64)
            primes = torch.tensor([1_000_003, 1_000_005], dtype=torch.int64, device=existing_pairs.device)
            A_hash = (existing_pairs * primes).sum(dim=1)
            B_hash = (unique_pairs.to(torch.int64) * primes).sum(dim=1)

            A_hash_sorted, sort_idx = torch.sort(A_hash)
            pos = torch.searchsorted(A_hash_sorted, B_hash)
            pos_clamped = pos.clamp(max=A_hash_sorted.shape[0] - 1)
            match = A_hash_sorted[pos_clamped] == B_hash
            matched_indices = sort_idx[pos_clamped[match]]

            new_mask = ~match
        else:
            matched_indices = torch.empty(0, dtype=torch.long, device=parent_indices.device)
            new_mask = torch.ones(unique_pairs.shape[0], dtype=torch.bool, device=parent_indices.device)

        return unique_pairs, inverse, matched_indices, new_mask
