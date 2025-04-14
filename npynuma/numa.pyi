from typing import Any, List

def get_numa_nodes() -> List[int]:
    """Get list of available NUMA node IDs.
    
    Returns:
        List of integer node IDs available on the system.
    
    Raises:
        RuntimeError: If NUMA subsystem is not available
    """
    ...

def set_numa_policy(node: int) -> 'PyDataMem_Handler':
    """Set NUMA memory allocation policy for numpy arrays.
    
    Args:
        node: NUMA node ID to allocate memory from. The process must have
            permission to allocate on the specified node.
    
    Returns:
        Previous memory handler object that can be used to restore
        the previous allocation policy.
    
    Raises:
        RuntimeError: If NUMA subsystem is not available
    """
    ...

def reset_numa_policy() -> None:
    """Reset numpy array allocation to default policy.
    
    This should be called when numa-allocated arrays are no longer needed
    to restore normal memory allocation behavior.
    """
    ...
