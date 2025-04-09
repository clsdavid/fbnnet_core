import numpy as np

import unittest
import fbnnet_core  # This is your C++-wrapped Python module


class TestCore(unittest.TestCase):
    def test_extract_gene_state_from_time_series_cube(self):
        cube1 = np.array([[1, 2], [3, 4]], dtype=np.int32)
        cube2 = np.array([[5, 6], [7, 8]], dtype=np.int32)
        # build a cube
        # The `axis=0` in `np.stack([cube1, cube2], axis=0)` specifies that the stacking should 
        # occur along a new first dimension (axis 0). This means the two cubes (`cube1` and `cube2`) 
        # will be stacked on top of each other, creating a higher-dimensional array.
        cube = np.stack([cube1, cube2], axis=0)
        result = fbnnet_core.extract_gene_state_from_time_series_cube(cube, 2)
        print(result)
        self.assertEqual(result.shape, (2, 6))
        
    def test_extract_gene_states(self):

        # Create sample data
        state_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
        row_names = ["gene1", "gene2", "gene3"]
        target_genes = ["gene3", "gene1"]

        # Call the function
        result = fbnnet_core.extract_gene_states(state_matrix, target_genes, row_names)

        # Access results
        filtered_matrix = result["matrix"]
        filtered_row_names = result["rownames"]

        print("Filtered matrix:")
        print(filtered_matrix)
        print("Row names:", filtered_row_names)
        self.assertEqual(filtered_matrix.shape, (2, 3))
        self.assertEqual(filtered_row_names, ['gene1', 'gene3'])
        # cube1 = np.array([[1, 2], [3, 4]], dtype=np.int32)
        # cube2 = np.array([[5, 6], [7, 8]], dtype=np.int32)
        # # build a cube
        # # The `axis=0` in `np.stack([cube1, cube2], axis=0)` specifies that the stacking should 
        # # occur along a new first dimension (axis 0). This means the two cubes (`cube1` and `cube2`) 
        # # will be stacked on top of each other, creating a higher-dimensional array.
        # cube = np.stack([cube1, cube2], axis=0)
        # result = fbnnet_core.extract_gene_states(cube, 2)
        # print(result)
        # self.assertEqual(result.shape, (2, 6))
        


if __name__ == "__main__":
    # convert an array into matrix
    # Example 1D array
    arr = np.array([1, 2, 3, 4])

    # Reshape into a 2x2 matrix
    matrix = arr.reshape(2, 2)

    # Example 2x2 matrix
    matrix = np.array([[1, 2], [3, 4]])

    # Convert back to a 1D array
    array = matrix.flatten()

    print(array)

    print(matrix)

    # Create individual matrices
    matrix1 = np.array([[1, 2], [3, 4]])
    matrix2 = np.array([[5, 6], [7, 8]])
    matrix3 = np.array([[9, 10], [11, 12]])

    # Stack matrices along a new axis to form a cube
    cube = np.stack([matrix1, matrix2, matrix3], axis=0)
    print(" printing the cube shape")
    # The result `(3, 2, 2)` from `cube.shape` indicates the dimensions of the 3D array (cube):

    # 1. **3**: The cube contains 3 matrices (or slices) along the first dimension (axis 0).
    # 2. **2**: Each matrix has 2 rows (second dimension, axis 1).
    # 3. **2**: Each matrix has 2 columns (third dimension, axis 2).

    # In summary, the cube is a 3D array with 3 matrices, and each matrix is of size 2x2.
    print(cube.shape)  # Output: (3, 2, 2)
    print(" printing the cube")
    print(cube)
    print(" printing the 2 cube")
    # Two 3D cubes
    cube1 = np.ones((2, 2, 2))  # Shape: (2, 2, 2)
    cube2 = np.zeros((2, 2, 2))  # Shape: (2, 2, 2)

    # Stack along axis=1
    result = np.stack([cube1, cube2], axis=1)
    # - `cube1` and `cube2` each have a shape of `(2, 2, 2)`.
    # - After stacking along `axis=0`, the resulting array has a new first dimension, making its shape `(2, 2, 2, 2)`.
    # - The first dimension (`2`) corresponds to the number of cubes being stacked.
    # - `cube1` and `cube2` each have a shape of `(2, 2, 2)`.
    # - After stacking along `axis=1`, the resulting array has a new second dimension, making its shape `(2, 2, 2, 2)`.
    # - The first dimension (`2`) corresponds to the number of "outermost groups" in the original cubes, and the second dimension (`2`) corresponds to the number of cubes being stacked.

    print(result.shape)  # Output: (2, 2, 2, 2)

    print("------------------------------------------")
    unittest.main()

        # self.assertEqual(fbnnet_core.add(-1, 1), 0)