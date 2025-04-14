# npynuma
NUMA node allocation for NumPy arrays on Linux

## Install System Requirements
```bash
# Ubuntu/Debian
sudo apt install libnuma-dev

# Fedora/CentOS
sudo dnf install numactl-devel

# Python package
pip install npynuma
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
