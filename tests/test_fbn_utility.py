import numpy as np

import unittest
import fbnnet_utils  # This is your C++-wrapped Python module

class TestMathBindings(unittest.TestCase):
    def test_mcbind(self):
        # Test the mcbind (matrix column bind) function
        arr = np.array([[1, 2], [3, 4]], dtype=np.double)
        combined = fbnnet_utils.mcbind(arr, arr)
        print(combined)

        self.assertEqual(combined.shape, (2, 4))
        # self.assertEqual(fbnnet_utils.add(-1, 1), 0)
    
    def test_mrbind(self):
        # Test the mrbind (matrix row bind) function
        arr = np.array([[1, 2], [3, 4]], dtype=np.double)
        combined = fbnnet_utils.mrbind(arr, arr)
        print(combined)

        self.assertEqual(combined.shape, (4, 2))
    def test_mpaste(self):

        arr2 = np.array(["GeneA", "GeneB"])
        result = fbnnet_utils.mpaste(arr2, "_")
        print(result)
        self.assertEqual(result, "GeneA_GeneB")

    def test_to_string(self):
        # Test the to_string function
        result = fbnnet_utils.to_string(14.55)
        print(result)
        self.assertEqual(result, "14.55")

    def test_concatenate(self):
        # Test the concatenate function
        arr1 = np.array(["1", "2", "3"])
        arr2 = np.array(["4,", "5", "6"])
        result = fbnnet_utils.concatenate(arr1, arr2)
        print(result)
        self.assertTrue(np.array_equal(result, ['1', '2', '3', '4,', '5', '6']))

    def test_concatenateI(self):
        # Test the concatenate function
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([4, 5, 6])

        result = fbnnet_utils.concatenateI(arr1, arr2)
        print(result)
        self.assertTrue(np.array_equal(result, [1, 2, 3, 4, 5, 6]))

    def test_concatenateN(self):
        # Test the concatenate function
        arr1 = np.array([1.45, 2.33, 0.333])
        arr2 = np.array([1.4, 2.5, 6.8])

        result = fbnnet_utils.concatenateN(arr1, arr2)
        print(result)
        self.assertTrue(np.array_equal(result, [1.45, 2.33, 0.333, 1.4, 2.5, 6.8]))

    def test_dround(self):
        # Test the dround function

        result = fbnnet_utils.dround(1.55555, 2)
        print(result)
        self.assertTrue(result, 1.56)

        result = fbnnet_utils.dround(1.555555555, 4)
        print(result)
        self.assertTrue(result, 1.5556)

    def test_isReallyNA(self):
        # Test the isReallyNA function
        result = fbnnet_utils.isReallyNA(np.nan)
        print(result)
        self.assertTrue(result, True)

    def test_countZeros(self):
        # Test the countZeros function
        arr = np.array([0, 1, 1, 0, 0])
        result = fbnnet_utils.countZeros(arr)
        print(result)
        self.assertEqual(result, 3)

    def test_vector_sort(self):
        # Test the vector_sort function
        arr = np.array(["3", "1", "2"])
        result = fbnnet_utils.vector_sort(arr, True)
        print(result)
        self.assertTrue(result, ["1", "2", "3"])

        result = fbnnet_utils.vector_sort(arr, False)
        print(result)
        self.assertTrue(result, ["3", "2", "1"])

    def test_convertStringIntoVector(self):
        # Test the convertStringIntoVector function
        result = fbnnet_utils.convertStringIntoVector("A,B,S", 1, False)
        print(result)
        self.assertTrue(np.array_equal(result, ["A", ",", "B", ",", "S"]))

        result = fbnnet_utils.convertStringIntoVector("A,B,S", 1, True)
        print(result)
        self.assertTrue(np.array_equal(result, ["a", ",", "b", ",", "s"]))

    def test_a_in_b(self):
        # Test the a_in_b function
        arr1 = np.array(["A", "B", "C"])
        arr2 = np.array(["B", "C", "D"])
        result = fbnnet_utils.a_in_b(arr1, arr2)
        print(result)
        self.assertTrue(np.array_equal(result, [False, True, True]))

        # # Test the a_in_b function
        # arr1 = np.array(["B", "B", "C"])
        # arr2 = np.array(["B", "C", "D"])
        # result = fbnnet_utils.a_in_b(arr1, arr2)
        # print(result)
        # self.assertTrue(np.array_equal(result, [False, True, True]))

    def test_a_in_b_index(self):
        # Test the a_in_b_index function
        arr1 = np.array(["A", "B", "C"])
        arr2 = np.array(["B", "C", "D"])
        result = fbnnet_utils.a_in_b_index(arr1, arr2)
        print(result)
        self.assertTrue(np.array_equal(result, [0, 1]))

    def test_resizel(self):
        # Test the resizel function
        arr = [1, 2, 3]
        result = fbnnet_utils.resizel(arr, 5)
        print(result)
        self.assertTrue(np.array_equal(result, [1, 2, 3, None, None]))

        result = fbnnet_utils.resizel(arr, 2)
        print(result)
        self.assertTrue(np.array_equal(result, [1, 2]))

    def test_orderByname(self):
        # Test the orderByname function
        dict = {"B": 2, "A": 1, "C": 3}
        arr = np.array(["A", "C", "B"])
        result = fbnnet_utils.orderByName(dict, arr)
        print(result)
        self.assertTrue(result, {"A": 1, "C": 3, "B": 2})


    def test_removeEmptyElement(self):
        # Test the removeEmptyElements function
        arr = ["A", "", "B", "C", ""]
        result = fbnnet_utils.removeEmptyElement(arr)
        print(result)
        self.assertTrue(result, ["A", "B", "C"])

    def test_subtractM(self):
        # Test the subtractM function
        arr1 = np.array([[1, 2], [3, 4,]])
        arr2 = np.array([1,1,])
        result = fbnnet_utils.subtractM(arr1, arr2)
        print(result)
        self.assertTrue(np.array_equal(result, [[0, 1], [2, 3]]))

        # Test the subtractM function
        arr1 = np.array([[1, 2, 2], [3, 4, 3]])
        arr2 = np.array([1,1,])
        result = fbnnet_utils.subtractM(arr1, arr2)
        print(result)
        # convert result to int
        result = result.astype(int)
        self.assertTrue(np.array_equal(result, [[0, 1, 1], [2, 3, 2]]))
        # test number
        arr1 = np.array([[2, 1, 2], [1, 2, 1], [2, 2, 2], [1, 2, 1]])
        arr2 = np.array([0, 1, 1, 1])
        result = fbnnet_utils.subtractM(arr1, arr2)
        print(result)
        self.assertTrue(np.array_equal(result, [[2, 1, 2], [0, 1, 0], [1, 1, 1], [0, 1, 0]]))

        # test binary operation
        arr1 = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 1, 0]])
        arr2 = np.array([0, 1, 1, 1])
        result = fbnnet_utils.subtractM(arr1, arr2)
        print(result)
        self.assertTrue(np.array_equal(result, [[1, 0, 1], [1, 0, 1], [0, 0, 1], [1, 0, 1]]))

    def test_matchCount(self):
        # Test the matchCount function, this is used to count the number of column matches with provided vector
        # 1, 0, 1     0     1 0 1
        # 0, 1, 0  -  1   = 1 0 1
        # 1, 1, 0     1     0 0 1
        # 0, 1, 0     1     1 0 1
        arr1 = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 1, 0]])
        arr2 = np.array([0, 1, 1, 1])
        result = fbnnet_utils.matchCount(arr1, arr2)
        print(result)
        self.assertEqual(result, 1)
        # p27:  0 1 0 1 9 0 1 0 1 9 1 1 0 
        # CycD: 1 1 1 9 1 1 0 1 9 1 0 0 1 

        state_matrix = np.array([[0, 1, 0, 1, 9, 0, 1, 0, 1, 9, 1, 1, 0 ],
            [1, 1,1, 9, 1, 1, 0, 1, 9, 1, 0, 0, 1 ]], dtype=np.float64)
        
        tt = np.array([1, 1])
        result2 = fbnnet_utils.matchCount(state_matrix, tt)
        print(result2)
        self.assertEqual(result2, 1)
        


    def test_compute_fisher_test(self):
        # Example 2x2 table (flattened as in R)
        table = np.array([10, 5, 20, 15], dtype=float)

        # Call the C++ function (internally uses statsmodels)
        result = fbnnet_utils.compute_fisher_test(table, conf_level=0.95)

        print("P-value:", result["p_value"])
        print("Odds ratio:", result["estimate"])
        print("95% CI:", result["conf_int"])
        self.assertTrue(result["p_value"] > 0.5287)
        self.assertTrue(result["estimate"] == 1.5)
        self.assertTrue(result["conf_int"][0] > 1.44)
        self.assertTrue(result["conf_int"][1] > 1.5)

    def test_compute_chisq(self):

        result = fbnnet_utils.compute_chisq(0.2, 0.3, 0.4, 0.1)
        print("Chi-squared test result:", result)
        # self.assertTrue(result["p_value"] > 0.05)
        # self.assertTrue(result["statistic"] > 0.05)




        # Call the C++ function (internally uses statsmodels)

    def test_subCPP(self):
        # Test the subCPP function
        arr1 = np.array(["A", "!", "B", "C"])
        arr2 = np.array(["A", "B", "!", "C"])
        arr3 = np.array(["A", "!", "B", "C"]) 
        result = fbnnet_utils.sub_cpp(arr1, arr2, arr3)
        print(result)
        self.assertTrue(np.array_equal(result, ["A", "B", "!", "C"]))

    def test_char_sort(self):
        # Test the char_sort function
        arr = np.array(["B", "A", "C"])
        result = fbnnet_utils.char_sort(arr, True)
        print(result)
        self.assertTrue(np.array_equal(result, ["C", "B", "A"]))

        result = fbnnet_utils.char_sort(arr, False)
        print(result)
        self.assertTrue(np.array_equal(result, ["A", "B", "C"]))

    def test_int_sort(self):
        # Test the int_sort function
        arr = np.array([3, 1, 2])
        result = fbnnet_utils.int_sort(arr, True)
        print(result)
        self.assertTrue(np.array_equal(result, [3, 2, 1]))

        result = fbnnet_utils.int_sort(arr, False)
        print(result)
        self.assertTrue(np.array_equal(result, [1, 2, 3]))


    def test_num_sort(self):
        # Test the num_sort function
        arr = np.array([3.1, 1.2, 2.3])
        result = fbnnet_utils.num_sort(arr, True)
        print(result)
        self.assertTrue(np.array_equal(result, [3.1, 2.3, 1.2]))

        result = fbnnet_utils.num_sort(arr, False)
        print(result)
        self.assertTrue(np.array_equal(result, [1.2, 2.3, 3.1]))

    def test_a_not_in_b(self):
        # Test the a_not_in_b function
        arr1 = np.array(["A", "B", "C"])
        arr2 = np.array(["B", "C", "D"])
        result = fbnnet_utils.a_not_in_b(arr1, arr2)
        print(result)
        self.assertTrue(np.array_equal(result, [True, False, False]))

    def test_a_not_in_b_index(self):
        # Test the a_not_in_b_index function
        arr1 = np.array(["A", "B", "C"])
        arr2 = np.array(["B", "C", "D"])
        result = fbnnet_utils.a_not_in_b_index(arr1, arr2)
        print(result)
        self.assertTrue(np.array_equal(result, [0]))

    def test_splitExpression(self):
        # Test the splitExpression function
        result1 = fbnnet_utils.splitExpression("A&B|C", 1)  # Type 1 splitting
        print(result1)
        self.assertTrue(np.array_equal(result1, ["A", "&", "B|C"]))
        result2 = fbnnet_utils.splitExpression("A&B|C", 2, True)  # Type 2 splitting with lowercase
        print(result2)
        self.assertTrue(np.array_equal(result1, ["A", "&", "B|C"]))

if __name__ == "__main__":
    unittest.main()
