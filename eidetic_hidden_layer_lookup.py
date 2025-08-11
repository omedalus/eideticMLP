import torch
from typing import List, Optional, Tuple


class EideticHiddenLayerLookup:
    def __init__(self):
        # Store (key, value) pairs
        self._storage: List[Tuple[List[float], List[float]]] = []
        self.insertion_exclusion_radius = 0.1
        self.lookup_exclusion_radius = 0.1
        self.enabled = True

    def __len__(self):
        """Return the number of stored key-value pairs."""
        return len(self._storage)

    def is_any_key_near(self, key: torch.Tensor, exclusion_radius: float) -> bool:
        keylist = [float(x) for x in key]
        dimensionality = len(keylist)
        for stored_key, _ in self._storage:
            dist = sum(abs(a - b) for a, b in zip(stored_key, keylist))
            if dist < (exclusion_radius * dimensionality):
                return True
        return False

    def insert(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> bool:
        """
        Insert a key-value pair if no existing key is within the exclusion radius.
        Returns True if the pair was inserted, False if it was blocked by another key
        already within the exclusion radius.
        key: 1D torch.Tensor
        value: torch.Tensor (any shape)
        """
        if not self.enabled:
            return False

        keylist = [float(x.detach()) for x in key]

        if self._storage:
            randomkey = self._storage[0][0]
            randomvalue = self._storage[0][1]
            if len(keylist) != len(randomkey):
                raise ValueError(
                    f"Key dimensionality must match. Existing key dimensionality: {len(randomkey)}, new key dimensionality: {len(keylist)}"
                )
            if len(value) != len(randomvalue):
                raise ValueError(
                    f"Value dimensionality must match. Existing value dimensionality: {len(randomvalue)}, new value dimensionality: {len(value)}"
                )

        valuelist = [float(x.detach()) for x in value]

        if self.is_any_key_near(keylist, self.insertion_exclusion_radius):
            return False

        self._storage.append((keylist, valuelist))
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
        if not self.enabled:
            return None
        if not self._storage:
            return None

        keylist = [float(x.detach()) for x in key]
        dimensionality = len(keylist)

        exclusion_dist = self.lookup_exclusion_radius * dimensionality

        min_dist = None
        nearest_value = None

        for stored_key, stored_value in self._storage:
            dist = sum(abs(a - b) for a, b in zip(stored_key, keylist))
            if dist < exclusion_dist:
                continue
            if (min_dist is None) or (dist < min_dist):
                min_dist = dist
                nearest_value = stored_value

        return nearest_value

    def lookup_batch(self, keys: torch.Tensor) -> Optional[torch.Tensor]:
        """
        For each key in the 2D tensor 'keys', call lookup and return a tensor of results.
        keys: 2D torch.Tensor (batch_size, key_dim)
        Returns: torch.Tensor of results stacked along the first dimension.
        """
        if not self._storage or not len(self._storage):
            return None

        randomvalue: List[float] = self._storage[0][1]
        valuedimensionality = len(randomvalue)

        # Prepare a results array. It needs to be the same dimensionality
        # and on the same device as the keys.
        results = keys.new_zeros((keys.size(0), valuedimensionality))

        for i in range(keys.size(0)):
            result = self.lookup(keys[i])
            if result is None:
                continue
            tensorresult = torch.tensor(result, dtype=keys.dtype)
            results[i] = tensorresult
        # Try to stack results if possible, else return as is
        try:
            return torch.cat(results, dim=0)
        except Exception:
            return results
