import unittest
from py.fbnnet_utils import fbn_data_reduction, similarity_between_matrix, check_similarity, generate_similar_report, dissolve
from py.fbnnet_utils import check_right_type_timeseries_data, check_numeric, check_probability_type_data, is_boolean_type_timeseries_data
from py.fbnnet_utils import output_genes, output_timeseries_based_on_genes
import pandas as pd
import numpy as np

class TestFbnnet_utils(unittest.TestCase):
    def test_run_fbnnet_utils(self):
        # Test fbn_data_reduction
        # Create sample data
        sample1 = pd.DataFrame({'t1': [1, 0], 't2': [0, 1]}, index=['gene1', 'gene2'])
        sample2 = pd.DataFrame({'t1': [1, 0], 't2': [0, 1]}, index=['gene1', 'gene2'])  # duplicate
        sample3 = pd.DataFrame({'t1': [0, 1], 't2': [1, 0]}, index=['gene1', 'gene2'])

        cube = [sample1, sample2, sample3]
        reduced = fbn_data_reduction(cube)
        print(f"Original cube size: {len(cube)}")
        print(f"Reduced cube size: {len(reduced)}")

        # Test similarity_between_matrix
        # Test cases
        m1 = np.array([[1, 2], [3, 4]])
        
        # Identical matrix (should be "verysimilar")
        print(similarity_between_matrix(m1, m1.copy(), 1))  
        # ['verysimilar', 1.0, 1]
        
        # Slightly different matrix
        m2 = np.array([[1, 2], [3, 5]])
        print(similarity_between_matrix(m1, m2, 2))  
        # ['similar', 0.75, 2] (3 out of 4 elements match)
        
        # Very different matrix
        m3 = np.array([[10, 20], [30, 40]])
        print(similarity_between_matrix(m1, m3, 3))  
        # ['veryunlikely', 0.0, 3]

        # Test check_similarity
    # Create test data
        original = [
            np.array([[1, 2], [3, 4]]),      # Sample 1 (will be identical)
            np.array([[5, 6], [7, 8]]),      # Sample 2 (1 value different)
            np.array([[9, 10], [11, 12]])    # Sample 3 (all values different)
        ]
        
        reconstructed = [
            np.array([[1, 2], [3, 4]]),      # Perfect match
            np.array([[5, 6], [7, 9]]),      # 1 value different (75% match)
            np.array([[90, 100], [110, 120]]) # Complete mismatch
        ]
        
        # Run similarity check
        similarity_report = check_similarity(original, reconstructed)
        
        # Print results
        print("Time Series Similarity Report:")
        print("-" * 40)
        for report in similarity_report:
            category, score, sample_id = report
            print(f"Sample {sample_id}:")
            print(f"  Similarity: {category}")
            print(f"  Match score: {score:.2f}")
            print(f"  Verdict: {'✅' if score > 0.6 else '❌'}")
            print("-" * 40)

        
        # Create test similarity report (matches R structure)
        test_report = [
            ["verysimilar", 0.95, 1],   # A
            ["similar", 0.85, 2],        # B
            ["similar", 0.75, 3],        # C
            ["likely", 0.65, 4],         # D
            ["likely", 0.55, 5],         # E
            ["unlikely", 0.45, 6],       # F
            ["unlikely", 0.35, 7],       # G
            ["veryunlikely", 0.25, 8],   # H
            ["veryunlikely", 0.15, 9],   # I
            ["veryunlikely", 0.05, 10]   # J
        ]
        
        # Generate the report
        organized_report = generate_similar_report(test_report)
        
        # Print the results (matches R's output format)
        print("Organized Similarity Report:")
        for category, entries in organized_report.items():
            print(f"\nCategory {category}:")
            for entry in entries:
                print(f"  Sample {entry[2]}: {entry[0]} (score: {entry[1]:.2f})")

        # Test dissolve function
        # Test with a complex nested structure similar to R's lists
        complex_list = {
            'group1': {
                'sub1': [10, 20, 30],
                'sub2': {'a': 100, 'b': 200}
            },
            'group2': 50,
            'group3': [{'x': 1}, {'y': 2}]
        }
        
        result = dissolve(complex_list)
        
        print("Original complex structure:")
        print(complex_list)
        
        print("\nDissolved structure:")
        for key, value in result.items():
            print(f"{key}: {value}")

        # Test check_right_type_timeseries_data
        try:
            test_data = [np.array([[1,2],[3,4]]), np.array([[5,6]])]
            check_right_type_timeseries_data(test_data)
            print("check_right_type_timeseries_data test passed")
        except ValueError as e:
            print(f"Test failed: {e}")
        
        # Test check_numeric
        try:
            check_numeric(5)
            check_numeric(3.14)
            print("check_numeric test passed")
        except ValueError as e:
            print(f"Test failed: {e}")
        
        # Test check_probability_type_data
        try:
            check_probability_type_data(0.5)
            print("check_probability_type_data test passed")
        except ValueError as e:
            print(f"Test failed: {e}")
        
        # Test is_boolean_type_timeseries_data
        bool_data = [np.array([[0,1],[1,0]]), np.array([[1,1]])]
        print(f"is_boolean_type_timeseries_data test: {is_boolean_type_timeseries_data(bool_data)}")  # Should be True
        
        # Test output_genes
        test_genes = ['Gene1', 'Gene2', 'Gene3']
        output_genes(test_genes, '.', 'test_genes.txt')
        print("output_genes test completed - check test_genes.txt")
        
        # Test output_timeseries_based_on_genes
        test_matrix = pd.DataFrame([[1,2],[3,4],[5,6]], index=['Gene2', 'Gene1', 'Gene3'])
        reordered = output_timeseries_based_on_genes(test_matrix, ['Gene1', 'Gene2', 'Gene3'])
        print("Reordered matrix:")
        print(reordered)

        print("=== Testing check_right_type_timeseries_data ===")
        test_data = [np.array([[1,2],[3,4]]), np.array([[5,6]])]
        try:
            check_right_type_timeseries_data(test_data)
            print("✓ Valid matrix list passed")
        except ValueError as e:
            print(f"✗ Failed: {e}")
        
        print("\n=== Testing check_numeric ===")
        try:
            check_numeric(3.14)
            print("✓ Valid number passed")
            check_numeric("text")
        except ValueError as e:
            print(f"✗ Failed as expected: {e}")
        
        print("\n=== Testing boolean check ===")
        bool_data = [np.array([[0,1],[1,0]])]
        print(f"Is boolean data? {is_boolean_type_timeseries_data(bool_data)}")
        
        print("\n=== Testing gene output ===")
        genes = ['TP53', 'BRCA1']
        output_genes(genes, '.', 'test_genes.txt')
        print("Gene list written to test_genes.txt")
        
        print("\n=== Testing matrix reordering ===")
        mat = pd.DataFrame([[1,2],[3,4]], index=['BRCA1', 'TP53'])
        print("Reordered matrix:")
        print(output_timeseries_based_on_genes(mat, ['TP53', 'BRCA1']))

        print("\n=== Testing matrix reordering ===")

        # Case 1: With pandas DataFrame
        df_mat = pd.DataFrame([[1,2],[3,4]], index=['BRCA1', 'TP53'])
        print("\nDataFrame input:")
        print(output_timeseries_based_on_genes(df_mat, ['TP53', 'BRCA1']))
        # Output:
        # [[3 4]
        #  [1 2]]

        # Case 2: With numpy array and explicit row names
        np_mat = np.array([[1,2],[3,4]])
        row_names = ['BRCA1', 'TP53']
        print("\nNumpy array with manual gene ordering:")
        print(output_timeseries_based_on_genes(np_mat, ['TP53', 'BRCA1']))
        # Output (same as input since no row names):
        # [[1 2]
        #  [3 4]]

        # Case 3: Invalid input
        try:
            output_timeseries_based_on_genes([[1,2],[3,4]], ['TP53'])
        except ValueError as e:
            print(f"\nExpected error: {e}")