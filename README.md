# npynuma
NUMA node allocation for NumPy arrays on Linux

## Install System Requirements
```bash
# Ubuntu/Debian
sudo apt install libnuma-dev

# Fedora/CentOS
sudo dnf install numactl-devel

# Python package
git clone https://github.com/xsgeng/npynuma
cd npynuma
pip install .

pip install npynuma # not yet
```

## Quick Start
```python
import numpy as np
from npynuma import NumaPolicy

with NumaPolicy(node=1):  # Allocate on NUMA node 1
    large_array = np.empty(10_000_000)  # 80MB array on node 1

# Works with existing numpy functions
with NumaPolicy(node=0):
    random_data = np.random.normal(size=1_000_000)
```

## Notes
- Requires Linux NUMA system (`numactl` installed)
- Node numbers must exist on your system

## Thread Safety
⚠️ **Not thread-safe** - The NUMA policy context is global to the process. 
Avoid using `NumaPolicy` contexts concurrently in:
- Multithreaded applications
- Async frameworks
- Any parallel code sharing NumPy allocations
