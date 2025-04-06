import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../build'))
import fbn_utils

arr = np.array([[1, 2], [3, 4]], dtype=np.double)
combined = fbn_utils.mcbind(arr, arr)
print(combined)