import torch
from typing import Optional

@torch.jit.script
def matching_row_indices(A: torch.Tensor, b: torch.Tensor, cols: torch.Tensor, validate_hash: bool = False) -> torch.Tensor:
    """ 
    Returns the row indices in A for which A[:, cols] == b[:, cols], using stable row-wise hashing.

    Assumes each row in b[:, cols] exists exactly once in A[:, cols].
    """
    A_cols = A[:, cols].to(torch.int64)
    B_cols = b[:, cols].to(torch.int64)
    
    primes = torch.tensor([1_000_003 + 2*i for i in range(A_cols.shape[1])], device=A.device, dtype=torch.int64)

    # Hash both sets
    A_hash = (A_cols * primes).sum(dim=1)
    B_hash = (B_cols * primes).sum(dim=1)

    # Sort A_hash for fast lookup
    A_hash_sorted, sort_indices = A_hash.sort()
    matching_pos = torch.searchsorted(A_hash_sorted, B_hash)

    # Validate hash matches
    if validate_hash:
        in_bounds = matching_pos < A_hash_sorted.shape[0]
        candidate_vals = A_hash_sorted[matching_pos.clamp(max=A_hash_sorted.shape[0] - 1)]
        valid_match = in_bounds & (candidate_vals == B_hash)

        assert valid_match.all(), "Hash mismatch: some b rows not found in A or collision occurred"

    # Return indices in original A
    matched_indices = sort_indices[matching_pos]
    return matched_indices

#@torch.jit.script
def matching_row_indices_all(
    A: torch.Tensor,
    b: torch.Tensor,
    cols: torch.Tensor,
) -> torch.Tensor:
    """
    Returns indices in A that match any row in b[:, cols] using fast hashing and searchsorted.

    Assumes each row in b[:, cols] appears at least once in A[:, cols].
    """
    A_cols = A[:, cols].to(torch.int64)
    B_cols = b[:, cols].to(torch.int64)

    d = A_cols.shape[1]
    primes = torch.tensor([1_000_003 + 2 * i for i in range(d)],
                          device=A.device, dtype=torch.int64)

    A_hash = (A_cols * primes).sum(dim=1)
    B_hash = (B_cols * primes).sum(dim=1)

    # Sort B_hash for binary search
    B_hash_sorted, _ = B_hash.sort()

    # For each A_hash, find if it exists in B_hash using searchsorted
    pos = torch.searchsorted(B_hash_sorted, A_hash)

    # Check if hash matched
    in_bounds = pos < B_hash_sorted.size(0)
    matched = in_bounds & (B_hash_sorted[pos.clamp(max=B_hash_sorted.size(0) - 1)] == A_hash)

    matched_indices = torch.nonzero(matched, as_tuple=True)[0]
    return matched_indices

#@torch.jit.script
def filter_rows(A: torch.Tensor, B: torch.Tensor, cols=slice(None)) -> torch.Tensor:
    """
    Filters out rows in B that are already present in A[:, cols], i.e., A[:, cols] == B[:, cols].

    Uses row-wise hashing with large primes for robustness.

    Args:
        A (torch.Tensor): Reference tensor of shape (n, d)
        B (torch.Tensor): Query tensor of shape (m, d)
        cols: Columns to compare (e.g., slice(0,2), or list like [0,2])

    Returns:
        filtered_B: B with rows NOT found in A        
    """
    if A.shape[0] == 0:
        return B#, torch.empty(0, dtype=torch.long, device=B.device)

    A_cols = A[:, cols].to(torch.int64)
    B_cols = B[:, cols].to(torch.int64)

    primes = torch.tensor([1_000_003 + 2*i for i in range(A_cols.shape[1])], device=A.device, dtype=torch.int64)

    A_hash = (A_cols * primes).sum(dim=1)
    B_hash = (B_cols * primes).sum(dim=1)

    A_hash_sorted, sort_indices = A_hash.sort()
    pos = torch.searchsorted(A_hash_sorted, B_hash)

    in_bounds = pos < A_hash_sorted.shape[0]
    candidate_vals = A_hash_sorted[pos.clamp(max=A_hash_sorted.shape[0] - 1)]
    found = in_bounds & (candidate_vals == B_hash)

    #if torch.any(found):
    #    assert (candidate_vals[found] == B_hash[found]).all(), "Hash mismatch: collision detected for some matched rows"

    #existing_indices = torch.nonzero(found, as_tuple=True)[0]
    filtered_B = B[~found]
    return filtered_B
    #return filtered_B, existing_indices