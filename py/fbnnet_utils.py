from typing import List, Dict, Any, Union
import pandas as pd

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


if __name__ == "__main__":

    import pandas as pd

    # Create sample data
    sample1 = pd.DataFrame({'t1': [1, 0], 't2': [0, 1]}, index=['gene1', 'gene2'])
    sample2 = pd.DataFrame({'t1': [1, 0], 't2': [0, 1]}, index=['gene1', 'gene2'])  # duplicate
    sample3 = pd.DataFrame({'t1': [0, 1], 't2': [1, 0]}, index=['gene1', 'gene2'])

    cube = [sample1, sample2, sample3]
    reduced = fbn_data_reduction(cube)
    print(f"Original cube size: {len(cube)}")
    print(f"Reduced cube size: {len(reduced)}")