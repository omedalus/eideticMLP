import torch
from typing import List, Optional, Tuple


class EideticHiddenLayerLookup:
    def __init__(self):
        # Store (key, value) pairs
        self._storage: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.insertion_exclusion_radius = 0.1
        self.lookup_exclusion_radius = 0.1

    def __len__(self):
        """Return the number of stored key-value pairs."""
        return len(self._storage)

    def insert(self, key: torch.Tensor, value: torch.Tensor) -> bool:
        """
        Insert a key-value pair if no existing key is within the exclusion radius.
        Returns True if the pair was inserted, False if it was blocked by another key
        already within the exclusion radius.
        key: 1D torch.Tensor
        value: torch.Tensor (any shape)
        """
        if not isinstance(key, torch.Tensor):
            raise ValueError(
                "Insertion key must be a 1D torch.Tensor. Instead it's {}".format(
                    type(key)
                )
            )
        if key.dim() != 1:
            raise ValueError(
                "Insertion key must be a 1D torch.Tensor. Instead it has {} dimensions".format(
                    key.dim()
                )
            )
        # Sanity check: all keys must be the same shape
        if self._storage:
            first_key_shape = self._storage[0][0].shape
            if key.shape != first_key_shape:
                raise ValueError(
                    f"All keys must have the same shape. Existing key shape: {first_key_shape}, new key shape: {key.shape}"
                )
            # Exclusion radius check
            for stored_key, _ in self._storage:
                dist = torch.sum(torch.abs(stored_key - key)).item()
                # The exclusion distance is scaled up by the dimensionality.
                if dist < (self.insertion_exclusion_radius * key.shape[0]):
                    # Do not insert if within exclusion radius
                    return False
        self._storage.append((key.clone(), value.clone()))
        return True

    def insert_batch(self, keys: torch.Tensor, values: torch.Tensor):
        """
        Insert multiple key-value pairs, skipping any whose key is within the exclusion radius of an existing key.
        keys: 2D torch.Tensor (batch_size, key_dim)
        values: torch.Tensor (batch_size, ...)
        """
        if not isinstance(keys, torch.Tensor):
            raise ValueError("insert_batch keys must be a torch.Tensor")
        if not isinstance(values, torch.Tensor):
            raise ValueError("insert_batch values must be a torch.Tensor")
        if keys.dim() != 2:
            raise ValueError(f"insert_batch keys must be 2D, got {keys.dim()}D")
        if values.size(0) != keys.size(0):
            raise ValueError(
                f"Number of keys ({keys.size(0)}) and values ({values.size(0)}) must match"
            )
        for i in range(keys.size(0)):
            self.insert(keys[i], values[i])

    def lookup(self, key: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Find the value whose key is nearest (Manhattan distance) to the given key,
        as long as it is not within the exclusion radius.
        Returns the value tensor.
        """
        if not self._storage:
            raise ValueError("No keys have been inserted.")
        if not isinstance(key, torch.Tensor):
            raise ValueError(
                "Lookup key must be a 1D torch.Tensor. Instead it's {}".format(
                    type(key)
                )
            )
        if key.dim() != 1:
            raise ValueError(
                "Lookup key must be a 1D torch.Tensor. Instead it has {} dimensions".format(
                    key.dim()
                )
            )
        min_dist = None
        nearest_value = None
        for stored_key, stored_value in self._storage:
            if stored_key.shape != key.shape:
                raise ValueError("All keys must have the same shape.")
            dist = torch.sum(torch.abs(stored_key - key)).item()
            if dist < self.lookup_exclusion_radius:
                # Ignore keys within the exclusion radius
                continue
            if (min_dist is None) or (dist < min_dist):
                min_dist = dist
                nearest_value = stored_value
        return nearest_value

    def lookup_batch(self, keys: torch.Tensor) -> torch.Tensor:
        """
        For each key in the 2D tensor 'keys', call lookup and return a tensor of results.
        keys: 2D torch.Tensor (batch_size, key_dim)
        Returns: torch.Tensor of results stacked along the first dimension.
        """
        if not isinstance(keys, torch.Tensor):
            raise ValueError("Lookup batch keys argument must be a torch.Tensor")
        if keys.dim() < 2:
            raise ValueError(
                f"Lookup batch keys argument must have at least 2 dimensions, got {keys.dim()}D"
            )
        results = []
        for i in range(keys.size(0)):
            result = self.lookup(keys[i])
            results.append(result.unsqueeze(0) if result.dim() > 0 else result)
        # Try to stack results if possible, else return as is
        try:
            return torch.cat(results, dim=0)
        except Exception:
            return results
