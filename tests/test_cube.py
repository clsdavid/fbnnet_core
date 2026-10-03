import numpy as np

import unittest
import fbnnet_core  # This is your C++-wrapped Python module
import fbnnet_utils
import fbnnet_matrix
from types import SimpleNamespace
import pandas as pd
from py_src.cube import construct_fbn_cube
from concurrent.futures import ThreadPoolExecutor
from py_src.cube import construct_fbn_cube


class TestCube(unittest.TestCase):
    def test_cube(self):
       # Example usage
        from py_src.boolnet import load_network
        from py_src.data_utils import generateAllCombinationBinary, generateBoolNetTimeseries
        # Write the network definition to a file
        with open("example.bn", "w") as f:
            f.write("targets, factors\n")
            f.write("Gene1, Gene1\n")
            f.write("Gene2, Gene1 & Gene5 & !Gene4\n")
            f.write("Gene3, Gene3\n")
            f.write("Gene4, Gene3 & !(Gene1 & Gene5)\n")
            f.write("Gene5, !Gene2\n")
        
        network = load_network("example.bn")
        print(network)
        initialStates = generateAllCombinationBinary(network["genes"])
        trainingseries = generateBoolNetTimeseries(network, initialStates, 43, transition_type = "synchronous")
        # convert numpy arrays to array of pandas DataFrames
        trainingseries = [pd.DataFrame(mat, index=network["genes"], columns=[str(j+1) for j in range(mat.shape[1])]) for mat in trainingseries]
        # output the keys of the network to list
        genes = list(network['genes'])
        # create timestampes to compare the results between parallel and non-parallel
        start_time = pd.Timestamp.now()
        result = construct_fbn_cube(genes, genes, trainingseries, max_k=5, temporal=1, use_parallel=False)
        end_time = pd.Timestamp.now()
        print(f"Time taken (non-parallel): {end_time - start_time}")
        # print(result)

        start_time = pd.Timestamp.now()
        result2 = construct_fbn_cube(genes, genes, trainingseries, max_k=5, temporal=1, use_parallel=True)
        end_time = pd.Timestamp.now()
        print(f"Time taken (parallel): {(end_time - start_time)}")
        # print(result)
        # Compare results
        if result == result2:
            print("Results are the same.")
        else:
            print("Results are different.")

        self.assertEqual(result, result2)
if __name__ == "__main__":

    unittest.main()

        # self.assertEqual(fbnnet_core.add(-1, 1), 0)