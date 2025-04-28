import multiprocessing
from typing import List, Dict, Any, Union
import numpy as np
import pandas as pd
import logging
import fbnnet_tree
import fbnnet_core
import fbnnet_matrix
from fbnnet_utils import fbn_data_reduction

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def construct_fbn_cube(target_genes: List[str], 
                      conditional_genes: List[str], 
                      timeseries_cube: List[pd.DataFrame], 
                      max_k: int = 5, 
                      temporal: int = 1, 
                      use_parallel: bool = False) -> Dict[str, Any]:
    """
    Create an FBN cube (Python implementation)
    
    This is the main function to generate a single FBN Cube or a group of cubes
    
    Parameters:
    -----------
    target_genes : List[str]
        A list of genes that will be treated as target genes
    conditional_genes : List[str]
        All genes that are available for building up the cube
    timeseries_cube : List[pd.DataFrame]
        A list of samples where each sample is a DataFrame that contains gene states
        (genes in rows and time points in columns)
    max_k : int, optional
        The maximum level the cube can dig in (default: 5)
    temporal : int, optional
        Number of previous steps the current one can depend on (default: 1)
    use_parallel : bool, optional
        If True, run in parallel, otherwise single-threaded (default: False)
    
    Returns:
    --------
    Dict[str, Any]
        An FBN cube that contains all precomputed measures
    """
    # Input validation
    if not isinstance(target_genes, list) or not all(isinstance(g, str) for g in target_genes):
        raise ValueError("The 'target_genes' must be a list of strings")
    if not target_genes:
        raise ValueError("The 'target_genes' cannot be empty")
    
    if not isinstance(conditional_genes, list) or not all(isinstance(g, str) for g in conditional_genes):
        raise ValueError("The 'conditional_genes' must be a list of strings")
    if not conditional_genes:
        raise ValueError("The 'conditional_genes' cannot be empty")
    
    if not isinstance(max_k, int) or max_k <= 0:
        raise ValueError("'max_k' must be a positive integer")
    
    if not isinstance(temporal, int) or temporal <= 0:
        raise ValueError("'temporal' must be a positive integer")
    
    logger.info(f"Enter construct_fbn_cube zone: "
               f"target_genes={len(target_genes)} genes and they are {', '.join(target_genes)}, "
               f"conditional_genes={len(conditional_genes)} genes and they are {', '.join(conditional_genes)}, "
               f"data_length={len(timeseries_cube)}, "
               f"max_k={max_k}, "
               f"temporal={temporal}, "
               f"use_parallel={use_parallel}")
    
    # Data reduction
    reduced_cube = fbn_data_reduction(timeseries_cube)
    genes_input = reduced_cube[0].index.tolist()
    for i, mat in enumerate(reduced_cube):
        reduced_cube[i] = fbnnet_matrix.FBNMatrix(mat, genes_input, [str(j+1) for j in range(mat.shape[1])])
    # Initialize state containers
    current_states = [None] * temporal
    previous_states = [None] * temporal
    current_states_c = [None] * temporal
    previous_states_c = [None] * temporal
    
    # Set up data for each temporal level
    for index in range(temporal, 0, -1):
        current_states[index-1] = fbnnet_core.extract_gene_state_from_timeseries_cube(reduced_cube, index)
        previous_states[index-1] = current_states[index-1]
        current_states_c[index-1] = current_states[index-1]
        previous_states_c[index-1] = current_states[index-1]
    
    # Calculate total timepoints and samples
    total_timepoints = sum(df.shape[1] for df in reduced_cube)
    total_samples = len(reduced_cube)
    all_gene_names = reduced_cube[0].index.tolist()
    
    # Create main parameters dictionary
    main_parameters = {
        'currentStates': current_states,
        'previousStates': previous_states,
        'currentStates_c': current_states_c,
        'previousStates_c': previous_states_c,
        'total_samples': total_samples,
        'all_gene_names': all_gene_names,
        'total_timepoints': total_timepoints
    }
    
    # Process each target gene
    if use_parallel:
        res = do_parallel_work(target_genes, conditional_genes, max_k, temporal, main_parameters)
    else:
        res = do_non_parallel_work(target_genes, conditional_genes, max_k, temporal, main_parameters)
    
    # Filter out None or empty results
    if res:
        res = [r for r in res if r is not None and len(r) > 0]
        # Add FBNCube class information (using dict to simulate R's class system)
        for item in res:
            item['class'] = 'FBNCube'
    
    logger.info("Leave construct_fbn_cube zone.")
    return res

# Helper functions (implementations would need to be added)



def do_parallel_work(target_genes: List[str], 
                    conditional_genes: List[str], 
                    max_k: int, 
                    temporal: int, 
                    main_parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Process genes in parallel"""
    with multiprocessing.Pool() as pool:
        results = pool.starmap(
            fbnnet_tree.process_cube_algorithm,
            [(gene, conditional_genes, max_k, temporal, main_parameters, None, None) for gene in target_genes]
        )
    return results

def do_non_parallel_work(target_genes: List[str], 
                        conditional_genes: List[str], 
                        max_k: int, 
                        temporal: int, 
                        main_parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Process genes sequentially"""
    results = []
    for gene in target_genes:
        results.append(
            fbnnet_tree.process_cube_algorithm(gene, conditional_genes, max_k, temporal, main_parameters, None, None)
        )
    return results