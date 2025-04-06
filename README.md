# fbnet_py

## test fbn_utility.cpp
``` bash
c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) fbn_utils.cpp -o fbn_utils$(python3-config --extension-suffix)
```

``` python
import numpy as np
import fbn_utils

arr = np.array([[1, 2], [3, 4]], dtype=np.double)
combined = fbn_utils.mcbind(arr, arr)
print(combined)
```