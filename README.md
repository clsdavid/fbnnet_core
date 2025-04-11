# fbnet_py

## test fbn_utility.cpp
``` bash
# install
pip install .

python setup.py clean --all
rm -rf build/ dist/ *.egg-info
pip uninstall fbnnet_core
pip install .

```

``` python
import numpy as np
import fbn_utils

arr = np.array([[1, 2], [3, 4]], dtype=np.double)
combined = fbn_utils.mcbind(arr, arr)
print(combined)
```