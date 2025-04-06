import numpy as np

import unittest
import fbnnet_core  # This is your C++-wrapped Python module

class TestMathBindings(unittest.TestCase):
    def test_mcbind(self):
        # Test the mcbind (matrix column bind) function
        arr = np.array([[1, 2], [3, 4]], dtype=np.double)
        combined = fbnnet_core.mcbind(arr, arr)
        print(combined)

        self.assertEqual(combined.shape, (2, 4))
        # self.assertEqual(fbnnet_core.add(-1, 1), 0)
    
    def test_mrbind(self):
        # Test the mrbind (matrix row bind) function
        arr = np.array([[1, 2], [3, 4]], dtype=np.double)
        combined = fbnnet_core.mrbind(arr, arr)
        print(combined)

        self.assertEqual(combined.shape, (4, 2))
    def test_mpaste(self):

        arr2 = np.array(["GeneA", "GeneB"])
        result = fbnnet_core.mpaste(arr2, "_")
        print(result)
        self.assertEqual(result, "GeneA_GeneB")

    def test_to_string(self):
        # Test the to_string function
        result = fbnnet_core.to_string(14.55)
        print(result)
        self.assertEqual(result, "14.55")

    def test_concatenate(self):
        # Test the concatenate function
        arr1 = np.array(["1", "2", "3"])
        arr2 = np.array(["4,", "5", "6"])
        result = fbnnet_core.concatenate(arr1, arr2)
        print(result)
        self.assertTrue(np.array_equal(result, ['1', '2', '3', '4,', '5', '6']))

    def test_concatenateI(self):
        # Test the concatenate function
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([4, 5, 6])

        result = fbnnet_core.concatenateI(arr1, arr2)
        print(result)
        self.assertTrue(np.array_equal(result, [1, 2, 3, 4, 5, 6]))

    def test_concatenateN(self):
        # Test the concatenate function
        arr1 = np.array([1.45, 2.33, 0.333])
        arr2 = np.array([1.4, 2.5, 6.8])

        result = fbnnet_core.concatenateN(arr1, arr2)
        print(result)
        self.assertTrue(np.array_equal(result, [1.45, 2.33, 0.333, 1.4, 2.5, 6.8]))

    def test_dround(self):
        # Test the dround function

        result = fbnnet_core.dround(1.55555, 2)
        print(result)
        self.assertTrue(result, 1.56)

        result = fbnnet_core.dround(1.555555555, 4)
        print(result)
        self.assertTrue(result, 1.5556)

    def test_isReallyNA(self):
        # Test the isReallyNA function
        result = fbnnet_core.isReallyNA(np.nan)
        print(result)
        self.assertTrue(result, True)

    def test_countZeros(self):
        # Test the countZeros function
        arr = np.array([0, 1, 1, 0, 0])
        result = fbnnet_core.countZeros(arr)
        print(result)
        self.assertEqual(result, 3)

if __name__ == "__main__":
    unittest.main()
