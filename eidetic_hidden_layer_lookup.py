import torch
from typing import List, Tuple


class EideticHiddenLayerLookup:
    def __init__(self):
        # Store (key, value) pairs
        self._storage: List[Tuple[torch.Tensor, torch.Tensor]] = []

    def insert(self, key: torch.Tensor, value: torch.Tensor):
        """
        Insert a key-value pair.
        key: 1D torch.Tensor
        value: torch.Tensor (any shape)
        """
        if not isinstance(key, torch.Tensor) or key.dim() != 1:
            raise ValueError("Key must be a 1D torch.Tensor")
        # Sanity check: all keys must be the same shape
        if self._storage:
            first_key_shape = self._storage[0][0].shape
            if key.shape != first_key_shape:
                raise ValueError(
                    f"All keys must have the same shape. Existing key shape: {first_key_shape}, new key shape: {key.shape}"
                )
        self._storage.append((key.clone(), value.clone()))

    def lookup(self, key: torch.Tensor) -> torch.Tensor:
        """
        Find the value whose key is nearest (Manhattan distance) to the given key.
        Returns the value tensor.
        """
        if not self._storage:
            raise ValueError("No keys have been inserted.")
        if not isinstance(key, torch.Tensor) or key.dim() != 1:
            raise ValueError("Key must be a 1D torch.Tensor")
        min_dist = None
        nearest_value = None
        for stored_key, stored_value in self._storage:
            if stored_key.shape != key.shape:
                raise ValueError("All keys must have the same shape.")
            dist = torch.sum(torch.abs(stored_key - key)).item()
            if (min_dist is None) or (dist < min_dist):
                min_dist = dist
                nearest_value = stored_value
        return nearest_value
