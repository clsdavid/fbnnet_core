from typing import List, Dict, Any, Union
import pandas as pd
import numpy as np
import os

def similarity_between_matrix(timeseries1, timeseries2, index):
    """
    Compare the similarity of two matrices
    
    Parameters:
    timeseries1 (numpy.ndarray): First matrix
    timeseries2 (numpy.ndarray): Second matrix to compare with
    index (int or str): A label to distinguish the result
    
    Returns:
    list: A vector showing similarity information about the two matrices
    
    Examples:
    >>> m1 = np.array([[1, 2], [3, 4]])
    >>> m2 = np.array([[1, 2], [3, 4]])
    >>> similarity_between_matrix(m1, m2, 1)
    ['verysimilar', 1.0, 1]
    """
    if timeseries1.shape != timeseries2.shape:
        raise ValueError("The two matrices must have the same dimensions")
    
    differ = np.abs(timeseries1 - timeseries2)
    
    # Calculate correlation as the percentage of zero differences
    zerosum = np.sum(differ == 0)
    correlation = zerosum / differ.size
    
    # Handle NA/None cases (though numpy won't typically produce these)
    if np.isnan(correlation) or correlation is None:
        correlation = 0
    
    # Determine similarity category
    if correlation <= 0.2:
        return ["veryunlikely", float(correlation), index]
    elif 0.2 < correlation <= 0.4:
        return ["unlikely", float(correlation), index]
    elif 0.4 < correlation <= 0.6:
        return ["likely", float(correlation), index]
    elif 0.6 < correlation <= 0.8:
        return ["similar", float(correlation), index]
    else:  # correlation > 0.8
        return ["verysimilar", float(correlation), index]
    
def fbn_data_reduction(timeseries_cube: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """
    Remove duplicate samples from a timeseries cube (Python implementation)
    
    This function mimics the R FBNDataReduction function which removes duplicated
    elements from the input list while preserving order.
    
    Parameters:
    -----------
    timeseries_cube : List[pd.DataFrame]
        A list of DataFrames where each DataFrame represents a timeseries sample
        (genes in rows, timepoints in columns)
    
    Returns:
    --------
    List[pd.DataFrame]
        A new list with duplicate samples removed (first occurrence is kept)
    """
    # Convert each DataFrame to a hashable tuple for duplicate checking
    seen = set()
    unique_cube = []
    
    for sample in timeseries_cube:
        # Convert the DataFrame to a tuple of tuples for hashability
        sample_tuple = tuple(tuple(row) for row in sample.values)
        
        if sample_tuple not in seen:
            seen.add(sample_tuple)
            unique_cube.append(sample)
    
    return unique_cube


def check_similarity(original_timeseries_cube, reconstructed_timeseries_cube):
    """
    Check the similarity between time series cubes
    
    Parameters:
    original_timeseries_cube (list of numpy arrays): Original data set
    reconstructed_timeseries_cube (list of numpy arrays): Reconstructed data set
    
    Returns:
    list: Similarity reports for each sample
    
    Examples:
    >>> original = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    >>> reconstructed = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 9]])]
    >>> report = check_similarity(original, reconstructed)
    >>> for r in report:
    ...     print(f"Sample {r[2]}: {r[0]} (score: {r[1]:.2f})")
    Sample 1: verysimilar (score: 1.00)
    Sample 2: similar (score: 0.75)
    """
    if len(original_timeseries_cube) != len(reconstructed_timeseries_cube):
        raise ValueError("The length of each timeseries data must be identical")
    
    results = []
    for i in range(len(original_timeseries_cube)):
        if original_timeseries_cube[i].shape != reconstructed_timeseries_cube[i].shape:
            raise ValueError("The dimension of each timeseries data must be identical")
        
        # Using i+1 to maintain 1-based indexing like R
        similarity_result = similarity_between_matrix(
            original_timeseries_cube[i],
            reconstructed_timeseries_cube[i],
            i+1
        )
        results.append(similarity_result)
    
    return results

def check_similarity(original_timeseries_cube, reconstructed_timeseries_cube):
    """
    Check the similarity between time series
    
    Parameters:
    original_timeseries_cube (list of numpy arrays): Original data set
    reconstructed_timeseries_cube (list of numpy arrays): Reconstructed data set
    
    Returns:
    list: similarity report for each sample
    
    Examples:
    >>> # Test with simple data
    >>> original = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    >>> reconstructed = [np.array([[1.1, 2.0], [3.0, 4.1]]), np.array([[5.2, 5.8], [7.1, 7.9]])]
    >>> report = check_similarity(original, reconstructed)
    >>> for r in report:
    ...     print(f"Sample {r['sample_index']}: Correlation = {r['correlation']:.4f}, MAE = {r['mean_absolute_error']:.4f}")
    Sample 1: Correlation = 0.9996, MAE = 0.0500
    Sample 2: Correlation = 0.9986, MAE = 0.1500
    """
    res = []
    
    if len(original_timeseries_cube) != len(reconstructed_timeseries_cube):
        raise ValueError("The length of each timeseries data must be identical")
    
    for i in range(len(original_timeseries_cube)):
        if original_timeseries_cube[i].shape != reconstructed_timeseries_cube[i].shape:
            raise ValueError("The dimension of each timeseries data must be identical")
        
        res.append(similarity_between_matrix(
            original_timeseries_cube[i], 
            reconstructed_timeseries_cube[i], 
            i+1  # Using 1-based indexing like R
        ))
    
    return res

def generate_similar_report(similarityreport):
    """
    Generate an organized similarity report
    
    Parameters:
    similarityreport (list): The raw similarity report created by check_similarity
    
    Returns:
    dict: An organized similarity report with categories A-J
    
    Examples:
    >>> report = [
    ...     ["verysimilar", 0.95, 1],
    ...     ["similar", 0.85, 2],
    ...     ["likely", 0.55, 3],
    ...     ["unlikely", 0.35, 4]
    ... ]
    >>> generate_similar_report(report)
    """
    # Initialize conditions
    cond1 = [entry for entry in similarityreport if float(entry[1]) >= 0.9]
    cond2 = [entry for entry in similarityreport if 0.8 <= float(entry[1]) < 0.9]
    cond3 = [entry for entry in similarityreport if 0.7 <= float(entry[1]) < 0.8]
    cond4 = [entry for entry in similarityreport if 0.6 <= float(entry[1]) < 0.7]
    cond5 = [entry for entry in similarityreport if 0.5 <= float(entry[1]) < 0.6]
    cond6 = [entry for entry in similarityreport if 0.4 <= float(entry[1]) < 0.5]
    cond7 = [entry for entry in similarityreport if 0.3 <= float(entry[1]) < 0.4]
    cond8 = [entry for entry in similarityreport if 0.2 <= float(entry[1]) < 0.3]
    cond9 = [entry for entry in similarityreport if 0.1 <= float(entry[1]) < 0.2]
    cond10 = [entry for entry in similarityreport if float(entry[1]) < 0.1]
    
    # Build the result dictionary
    res = {
        "A": cond1,
        "B": cond2,
        "C": cond3,
        "D": cond4,
        "E": cond5,
        "F": cond6,
        "G": cond7,
        "H": cond8,
        "I": cond9,
        "J": cond10
    }
    
    return res


def dissolve(x):
    """
    A function that moves sub-list items to its parent list, i.e. it flattens
    arbitrarily nested lists/tuples into a single flat list.

    This mirrors R's `dissolve` from utility_FBN.R, which only recurses into
    R lists (`is.list(x)`); R's named atomic vectors (the equivalent of a
    Python dict record, e.g. a single mined FBN rule) are treated as opaque
    leaf values and are *not* taken apart. So here only `list`/`tuple` values
    are recursed into - dicts (and any other scalar) are appended as-is.

    Parameters:
    x (list/tuple/Any): A (possibly nested) list/tuple, or a single leaf value

    Returns:
    list: A flat list of leaf values (dicts, strings, numbers, etc.)

    Examples:
    >>> dissolve([[{'a': 1}, {'b': 2}], {'c': 3}])
    [{'a': 1}, {'b': 2}, {'c': 3}]
    """
    combi = []

    def operator(current):
        if isinstance(current, (list, tuple)):
            for item in current:
                operator(item)
        elif current is not None:
            combi.append(current)

    operator(x)
    return combi

def check_right_type_timeseries_data(timeseries_data):
    """
    Check whether the data is the right type for FBNNet
    
    Args:
        timeseries_data: The timeseries data to check
        
    Raises:
        ValueError: If data is not the correct type
    """
    if not isinstance(timeseries_data, list):
        raise ValueError("The type of timeseries_data must be LIST")
    
    for item in timeseries_data:
        if not isinstance(item, np.ndarray) or len(item.shape) != 2:
            raise ValueError("The element of the data must be a matrix")

def check_numeric(x):
    """
    Check if a value is numeric
    
    Args:
        x: Value to check
        
    Raises:
        ValueError: If input is not numeric
    """
    if not isinstance(x, (int, float, np.number)):
        raise ValueError("The input is not a type of numeric")

def check_probability_type_data(x):
    """
    Check if a value is a probability between 0 and 1
    
    Args:
        x: Value to check
        
    Raises:
        ValueError: If input is not a valid probability
    """
    if not isinstance(x, (int, float, np.number)) or x < 0 or x > 1:
        raise ValueError("The input is not a type of probability or a value between 0 and 1")

def is_boolean_type_timeseries_data(x):
    """
    Check if timeseries data contains only binary values (0 and 1)
    
    Args:
        x: Timeseries data to check
        
    Returns:
        bool: True if all values are binary, False otherwise
    """
    for mat in x:
        unique_values = np.unique(mat)
        if not np.all(np.isin(unique_values, [0, 1])):
            return False
    return True

def output_annotated_genes(genes, path, filename, david_gene_list):
    """
    Convert a vector of gene names to annotated gene details and save to CSV
    
    Args:
        genes: List of gene names
        path: Output directory path
        filename: Output filename
        david_gene_list: DataFrame containing gene annotations
    """
    mapped_genes = david_gene_list[david_gene_list['Symbol'].isin(genes)]
    distinct_mapped_genes = mapped_genes.drop_duplicates(subset=['Symbol'], keep='first')
    distinct_mapped_genes.to_csv(os.path.join(path, filename), index=False)

def output_genes(genes, path, filename):
    """
    Output a vector of gene names separated by comma
    
    Args:
        genes: List of gene names
        path: Output directory path
        filename: Output filename
    """
    vect_str = ', '.join(str(gene) for gene in genes)
    with open(os.path.join(path, filename), 'w') as f:
        f.write(vect_str + '\n')

def output_timeseries_based_on_genes(matrix, genes):
    """
    Reorder a matrix by a vector of genes
    
    Args:
        matrix: Input matrix (2D numpy array with row names)
        genes: List of target genes matching matrix row names
        
    Returns:
        numpy.ndarray: Reordered matrix
        
    Raises:
        ValueError: If input is not a matrix or genes don't match
    """
    # Convert to numpy array if DataFrame is passed
    if isinstance(matrix, pd.DataFrame):
        matrix_values = matrix.values
        row_names = matrix.index.tolist()
    elif isinstance(matrix, np.ndarray):
        matrix_values = matrix
        row_names = None
    else:
        raise ValueError("The parameter matrix must be a matrix object (numpy array or pandas DataFrame)")
    
    # Check gene names exist if row names were provided
    if row_names and not all(gene in row_names for gene in genes):
        raise ValueError("All of the genes must be in the row names of the matrix")
    
    # Reorder if row names exist
    if row_names:
        gene_indices = [row_names.index(gene) for gene in genes]
        return matrix_values[gene_indices]
    else:
        return matrix_values  # Return as-is if no row names

if __name__ == "__main__":

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
    # Test with a nested list structure (dissolve only recurses into lists/tuples)
    complex_list = [[10, 20, 30], [{'a': 100, 'b': 200}], 50, [{'x': 1}, {'y': 2}]]

    result = dissolve(complex_list)

    print("Original complex structure:")
    print(complex_list)

    print("\nDissolved structure:")
    for value in result:
        print(value)

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