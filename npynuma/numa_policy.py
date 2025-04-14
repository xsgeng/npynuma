import numpy as np
from .numa import set_numa_policy, reset_numa_policy

class NumaPolicy:
    """
    Context manager to set NUMA policy for a block of code.
    """
    def __init__(self, node: int):
        """
        Args:
            node: NUMA node to set policy on. If None, no policy is set.
        """
        self.node = node
        self.old_handler = None

    def __enter__(self):
        if self.node is not None:
            self.old_handler = set_numa_policy(self.node)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.old_handler is not None:
            reset_numa_policy()