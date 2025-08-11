import torch
import numpy as np
from annoy import AnnoyIndex
from typing import List, Tuple, Optional


class EideticHiddenLayerLookup:
    def __init__(self, key_dim: Optional[int] = None, n_trees: int = 10):
        """
        key_dim: dimension of the key vectors (required for Annoy)
        n_trees: number of trees for Annoy index (higher = more accurate, slower build)
        """
        self.key_dim = key_dim
        self.n_trees = n_trees
        self._annoy_index = None
        self._values: List[torch.Tensor] = []
        self._built = False

    def __len__(self):
        """Return the number of stored key-value pairs."""
        return len(self._values)

    def insert(self, key: torch.Tensor, value: torch.Tensor):
        """
        Insert a key-value pair.
        key: 1D torch.Tensor
        value: torch.Tensor (any shape)
        """
        if not isinstance(key, torch.Tensor):
            raise ValueError(
                f"Insertion key must be a 1D torch.Tensor. Instead it's {type(key)}"
            )
        if key.dim() != 1:
            raise ValueError(
                f"Insertion key must be a 1D torch.Tensor. Instead it has {key.dim()} dimensions"
            )
        if self.key_dim is None:
            self.key_dim = key.shape[0]
        if key.shape[0] != self.key_dim:
            raise ValueError(
                f"All keys must have the same dimension {self.key_dim}, got {key.shape[0]}"
            )
        if self._annoy_index is None:
            self._annoy_index = AnnoyIndex(self.key_dim, "euclidean")
        idx = len(self._values)
        self._annoy_index.add_item(idx, key.cpu().numpy().tolist())
        self._values.append(value.clone())
        self._built = False

    def insert_batch(self, keys: torch.Tensor, values: torch.Tensor):
        """
        Insert multiple key-value pairs.
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

    def _ensure_built(self):
        if self._annoy_index is not None and not self._built:
            self._annoy_index.build(self.n_trees)
            self._built = True

    def lookup(self, key: torch.Tensor) -> torch.Tensor:
        """
        Find the value whose key is nearest (Euclidean distance) to the given key using Annoy.
        Returns the value tensor.
        """
        if not self._values:
            raise ValueError("No keys have been inserted.")
        if not isinstance(key, torch.Tensor):
            raise ValueError(
                f"Lookup key must be a 1D torch.Tensor. Instead it's {type(key)}"
            )
        if key.dim() != 1:
            raise ValueError(
                f"Lookup key must be a 1D torch.Tensor. Instead it has {key.dim()} dimensions"
            )
        if key.shape[0] != self.key_dim:
            raise ValueError(
                f"Lookup key must have dimension {self.key_dim}, got {key.shape[0]}"
            )
        self._ensure_built()
        idx = self._annoy_index.get_nns_by_vector(key.cpu().numpy().tolist(), 1)[0]
        return self._values[idx]

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
        try:
            return torch.cat(results, dim=0)
        except Exception:
            return results
