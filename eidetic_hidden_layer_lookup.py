import torch
from typing import Any, Dict, List, Optional, Tuple


class EideticHiddenLayerLookup:
    def __init__(self):
        # Store (key, value) pairs
        self.enabled = True

        self._key_dimensionality = 0
        self._dim_cellularization = 4
        self._storage: Dict[str, List[Any]] = {}
        self._num_items = 0

    @property
    def key_dimensionality(self) -> int:
        return self._key_dimensionality

    @property
    def cellularization(self) -> int:
        return self._dim_cellularization

    def __len__(self):
        """Return the number of stored key-value pairs."""
        return self._num_items

    def _cellularize_key(self, key: List[float]) -> List[int]:
        """
        Cellularize the key into a list of integers based on the cellularization.
        """
        retval = [int(x / self._dim_cellularization) for x in key]
        return retval

    def _stringify_key(self, key: List[float]) -> str:
        """
        Convert this key into a hashable string.
        """
        cellular_key = self._cellularize_key(key)
        keystr = "-".join([f"{x}" for x in cellular_key])
        return keystr

    def insert(self, key: List[float], value: any) -> bool:
        """
        Insert a key-value pair if no existing key is within the exclusion radius.
        Returns True if the pair was inserted, False if it was blocked by another key
        already within the exclusion radius.
        key: List[float]
        value: any (any shape)
        """
        if not self.enabled:
            return False

        if not self._key_dimensionality:
            self._key_dimensionality = len(key)

        if self._key_dimensionality != len(key):
            raise ValueError(
                f"Inconsistent key dimensionality. New key length is {len(key)}, but expected {self._key_dimensionality}."
            )

        keyhash = self._stringify_key(key)
        if keyhash not in self._storage:
            self._storage[keyhash] = []

        self._storage[keyhash].append(value)
        self._num_items += 1
        return True

    def lookup(self, key: List[float]) -> List[Any]:
        if not self.enabled:
            return []

        keyhash = self._stringify_key(key)
        retval = self._storage.get(keyhash, [])
        return retval

    def diagnostic_print(self):
        # Count which keys have the most elements.
        keyswithcount = [[k, len(v)] for k, v in self._storage.items() if len(v) > 10]
        for k, c in keyswithcount:
            print(k, c)
