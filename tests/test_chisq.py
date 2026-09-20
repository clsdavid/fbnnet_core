import numpy as np
from scipy.stats import chisquare
import unittest
import fbnnet_chisq  # This is your C++-wrapped Python module
import timeit

class TestChiSQ(unittest.TestCase):
    def test_contingency_chisq(self):
        # Test with a 2x2 contingency table
        table = np.array([
            [100, 50],   # Group 1
            [75, 75]     # Group 2
        ], dtype=np.int32)

        chi_value, expected = fbnnet_chisq.contingency_chisq(table)
        print("Chi-square test results:")
        print(f"Chi-square value: {chi_value:.4f}")
        print("Expected table:")
        print(expected)

        # Verify with scipy
        from scipy.stats import chi2_contingency
        scipy_chi = chi2_contingency(table)
        print("\nScipy results:")
        print(f"Chi-square: {scipy_chi[0]:.4f}")
        print("p-value:", scipy_chi[1])
        print("Expected table:")
        print(scipy_chi[3])

        # Test independence in a 2x2 contingency table
        table = np.array([
            [100, 50],   # Group 1
            [75, 75]     # Group 2
        ], dtype=np.int32)

        chi_value, expected = fbnnet_chisq.contingency_chisq(table)

        print(f"Contingency chi-square: {chi_value:.4f}")
        print("Expected table:")
        print(expected)

    def test_goodness_of_fit_chisq(self):
        # Test with a goodness-of-fit example
        observed_die = np.array([15, 12, 18, 17, 14, 14], dtype=np.float64)
        chi_value, expected = fbnnet_chisq.uniform_chisq(observed_die)

        print(f"goodness of fit Chi-square value: {chi_value:.4f}")
        print("Expected values:", expected)

    def test_possition_distribution_chisq(self):
        # Test with a position distribution example
        # Test if data follows Poisson distribution
        counts = np.array([12, 36, 42, 30, 18, 7, 3], dtype=np.int32)  # Counts of events
        lambda_est = 2.0  # Estimated lambda parameter
        chi_value, expected = fbnnet_chisq.poisson_chisq(counts, lambda_est)

        print(f"Poisson test chi-square: {chi_value:.4f}")
        print("Expected counts:", expected)

    def test_edge_cases(self):
        # Zero expected value (should raise)
        try:
            fbnnet_chisq.chisq_statistic(np.array([1.0]), np.array([0.0]))
        except ValueError as e:
            print("Correctly caught zero expected value:", e)
        
        # Unequal lengths (should raise)
        try:
            fbnnet_chisq.chisq_statistic(np.array([1.0, 2.0]), np.array([1.0]))
        except ValueError as e:
            print("Correctly caught unequal lengths:", e)
        
        # Empty arrays
        print("Empty arrays:", fbnnet_chisq.chisq_statistic(np.array([]), np.array([])))
        
        # Very small values
        small = np.array([1e-10, 2e-10], dtype=np.float64)
        print("Small values:", fbnnet_chisq.chisq_statistic(small, small*2))

    def test_chisq(self):
        # Example 1: Testing a fair die (small difference)
        obs1 = np.array([15, 12, 18, 17, 14, 14], dtype=np.float64)
        exp1 = np.array([15, 15, 15, 15, 15, 15], dtype=np.float64)
        
        # Example 2: Clear bias (large difference)
        obs2 = np.array([30, 10, 5, 5, 10, 30], dtype=np.float64)
        exp2 = exp1.copy()
        
        # Example 3: Larger dataset
        obs3 = np.random.poisson(100, size=100).astype(np.float64)
        exp3 = np.full(100, 100.0, dtype=np.float64)
        
        # Calculate chi-square statistics
        print("Fair die test:", fbnnet_chisq.chisq_statistic(obs1, exp1))
        print("Biased die test:", fbnnet_chisq.chisq_statistic(obs2, exp2))
        print("Large Poisson test:", fbnnet_chisq.chisq_statistic(obs3, exp3))
        
        # Compare with numpy/scipy implementation
        from scipy.stats import chisquare
        print("\nScipy verification:")
        print("Fair die:", chisquare(obs1, exp1).statistic)
        print("Biased die:", chisquare(obs2, exp2).statistic)
        try:
            print("Large Poisson:", chisquare(obs3, exp3).statistic)
        except ValueError as e:
            print("Caught error for large Poisson test:", e)


def benchmark():
    # Test with different array sizes
    sizes = [100, 1000, 10000, 100000]
    
    print("Size\tC++ (μs)\tPython (μs)\tSpeedup")
    print("----\t--------\t----------\t-------")
    
    for size in sizes:
        setup = f"""
import numpy as np
import fbnnet_chisq
obs = np.random.randint(1, 100, size={size})
exp = np.random.randint(1, 100, size={size})
"""
        
        # Time C++ version
        cpp_timer = timeit.Timer('fbnnet_chisq.chisq_statistic(obs, exp)', setup=setup)
        cpp_time = min(cpp_timer.repeat(5, 1000)) / 1000 * 1e6  # μs per loop
        
        # Time Python version
        py_timer = timeit.Timer('sum((o-e)**2/e for o,e in zip(obs,exp))', setup=setup)
        py_time = min(py_timer.repeat(5, 100)) / 100 * 1e6  # Fewer loops for slow Python
        
        print(f"{size}\t{cpp_time:.1f}\t\t{py_time:.1f}\t\t{py_time/cpp_time:.1f}x")

if __name__ == "__main__":
    # Benchmarking
    # Setup code that runs once
    setup = """
import numpy as np
import fbnnet_chisq  # Your C++ module
obs = np.random.randint(10, 100, size=1000)
exp = np.random.randint(10, 100, size=1000)
    """

    # Time C++ implementation
    cpp_time = timeit.timeit('fbnnet_chisq.chisq_statistic(obs, exp)', 
                            setup=setup, 
                            number=10000)

    # Time Python implementation
    py_time = timeit.timeit('sum((o-e)**2/e for o,e in zip(obs,exp))', 
                        setup=setup, 
                        number=10000)

    print(f"\nBenchmark Results (1000 elements, 10,000 iterations)")
    print(f"C++ version: {cpp_time:.4f} seconds total, {cpp_time/10000:.6f} per loop")
    print(f"Python version: {py_time:.4f} seconds total, {py_time/10000:.6f} per loop")
    print(f"\nSpeedup factor: {py_time/cpp_time:.1f}x")

    print("\n Run full benchmarking")
    benchmark()
    # Run unit tests
    print("\nRunning unit tests...")
    unittest.main()