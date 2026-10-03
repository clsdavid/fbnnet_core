import multiprocessing
import os
from typing import List, Dict, Any, Optional, Union
import numpy as np
import pandas as pd
import logging
import fbnnet_tree
import fbnnet_core
import fbnnet_matrix
from .general_utils import fbn_data_reduction


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# "cores - 1" convention: leave one core free for the OS/orchestrator process.
DEFAULT_MAX_WORKERS = max(1, (os.cpu_count() or 2) - 1)

_FORK_AVAILABLE = "fork" in multiprocessing.get_all_start_methods()

# Populated in the parent process right before a fork-context Pool is created.
# Forked workers inherit this via copy-on-write memory, so the (potentially
# unpicklable) pybind11 main_parameters object is never pickled across the
# process boundary -- only the per-task gene name is.
_cube_worker_state: Dict[str, Any] = {}


def _cube_worker(gene: str) -> Any:
    """Runs in a forked worker process; reads shared state via COW memory."""
    state = _cube_worker_state
    return fbnnet_tree.process_cube_algorithm(
        gene, state["conditional_genes"], state["max_k"], state["temporal"],
        state["main_parameters"], None, None,
    )

def convert_df_main_parameters(timeseries_cube: List[pd.DataFrame], temporal: int = 1) -> Dict[str, Any]:
    """ Convert a list of pandas DataFrames into a main parameters dictionary for FBN cube construction.
    This function processes the input time series data, reduces it, and prepares the necessary parameters
    for further analysis in the FBN cube construction."""
    # Data reduction
    reduced_cube = fbn_data_reduction(timeseries_cube)
    # get first matrix's row names
    genes_input = reduced_cube[0].index.tolist()
    # Convert each DataFrame to numpy array
    for i, mat in enumerate(reduced_cube):
        reduced_cube[i] = mat.to_numpy(dtype=np.float64)

    for i, mat in enumerate(reduced_cube):
        reduced_cube[i] = fbnnet_matrix.FBNMatrix(mat, genes_input, [str(j+1) for j in range(mat.shape[1])])
    # Initialize state containers
    current_states = [None] * temporal
    previous_states = [None] * temporal
    current_states_c = [None] * temporal
    previous_states_c = [None] * temporal
    
    # Set up data for each temporal level
    for index in range(temporal, 0, -1):
        current_states[index-1] = fbnnet_core.extract_gene_state_from_time_series_cube(reduced_cube, index)
        previous_states[index-1] = current_states[index-1]
        current_states_c[index-1] = current_states[index-1]
        previous_states_c[index-1] = current_states[index-1]
    
    # Calculate total timepoints and samples
    total_timepoints = sum(mat.matrix_t().shape[1] for mat in reduced_cube)
    total_samples = len(reduced_cube)
    all_gene_names = genes_input
    
    # Create main parameters dictionary
    main_parameters = {
        "currentStates": current_states,
        "previousStates": previous_states,
        "currentStates_c": current_states_c,
        "previousStates_c": previous_states_c,
        "total_samples": total_samples,
        "rownames": all_gene_names,
        "total_timepoints": total_timepoints,
        "testseries": reduced_cube
    }

    return main_parameters

def construct_fbn_cube(target_genes: List[str], 
                      conditional_genes: List[str], 
                      timeseries_cube: List[pd.DataFrame], 
                      max_k: int = 5, 
                      temporal: int = 1, 
                      use_parallel: bool = False,
                      max_workers: Optional[int] = None) -> Dict[str, Any]:
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
        If True, run target genes across separate OS processes (default: False)
    max_workers : int, optional
        Maximum number of worker processes when use_parallel is True
        (default: cpu_count - 1)
    
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
    
    # # Data reduction
    # reduced_cube = fbn_data_reduction(timeseries_cube)
    # # get first matrix's row names
    # genes_input = reduced_cube[0].index.tolist()
    # # Convert each DataFrame to numpy array
    # for i, mat in enumerate(reduced_cube):
    #     reduced_cube[i] = mat.to_numpy(dtype=np.float64)

    # for i, mat in enumerate(reduced_cube):
    #     reduced_cube[i] = fbnnet_matrix.FBNMatrix(mat, genes_input, [str(j+1) for j in range(mat.shape[1])])
    # # Initialize state containers
    # current_states = [None] * temporal
    # previous_states = [None] * temporal
    # current_states_c = [None] * temporal
    # previous_states_c = [None] * temporal
    
    # # Set up data for each temporal level
    # for index in range(temporal, 0, -1):
    #     current_states[index-1] = fbnnet_core.extract_gene_state_from_time_series_cube(reduced_cube, index)
    #     previous_states[index-1] = current_states[index-1]
    #     current_states_c[index-1] = current_states[index-1]
    #     previous_states_c[index-1] = current_states[index-1]
    
    # # Calculate total timepoints and samples
    # total_timepoints = sum(mat.matrix_t().shape[1] for mat in reduced_cube)
    # total_samples = len(reduced_cube)
    # all_gene_names = genes_input
    
    # # Create main parameters dictionary
    # main_parameters = {
    #     "currentStates": current_states,
    #     "previousStates": previous_states,
    #     "currentStates_c": current_states_c,
    #     "previousStates_c": previous_states_c,
    #     "total_samples": total_samples,
    #     "rownames": all_gene_names,
    #     "total_timepoints": total_timepoints,
    #     "testseries": reduced_cube
    # }
    main_parameters = convert_df_main_parameters(timeseries_cube, temporal)
    target_genes = fbnnet_tree.filterTargetGenesByConditionGenes(target_genes, main_parameters, conditional_genes, None, temporal)
    # Process each target gene
    if use_parallel:
        res = do_parallel_work(target_genes, conditional_genes, max_k, temporal, main_parameters, max_workers)
    else:
        res = do_non_parallel_work(target_genes, conditional_genes, max_k, temporal, main_parameters)
    
    # Filter out None or empty results
    cube = {}
    result = {}
    if res:
        res = [r for r in res if r is not None and len(r) > 0]
        # Add FBNCube class information (using dict to simulate R's class system)
        for item in res:
            cube.update(item)
    result["cube"] = cube
    result["class"] = "FBNCube"
    result["target_genes"] = target_genes
    logger.info("Leave construct_fbn_cube zone.")
    return result

# Helper functions (implementations would need to be added)



def do_parallel_work(target_genes: List[str], 
                     conditional_genes: List[str], 
                     max_k: int, 
                     temporal: int, 
                     main_parameters: Dict[str, Any],
                     max_workers: Optional[int] = None) -> List[Dict[str, Any]]:
    """Process genes in parallel using separate OS processes (fork), not threads.

    The pybind11-based main_parameters object is never pickled: it's stashed in a
    module-level global before the fork-context Pool is created, so forked workers
    inherit it via copy-on-write memory. Only the per-task gene name (a plain
    string) is pickled through the task queue. This replaces an earlier
    ThreadPoolExecutor-based implementation that provided no real speedup because
    none of the pybind11 bindings release the GIL.
    """
    global _cube_worker_state

    n_workers = max(1, min(max_workers or DEFAULT_MAX_WORKERS, len(target_genes)))
    if not _FORK_AVAILABLE or n_workers <= 1:
        if not _FORK_AVAILABLE:
            logger.warning(
                "multiprocessing 'fork' start method unavailable on this platform; "
                "falling back to sequential execution."
            )
        return do_non_parallel_work(target_genes, conditional_genes, max_k, temporal, main_parameters)

    _cube_worker_state = {
        "conditional_genes": conditional_genes,
        "max_k": max_k,
        "temporal": temporal,
        "main_parameters": main_parameters,
    }
    try:
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(processes=n_workers) as pool:
            results = pool.map(_cube_worker, target_genes)
    finally:
        _cube_worker_state = {}

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

if __name__ == "__main__":
    # Example usage
    from boolnet import load_network
    from data_utils import generateAllCombinationBinary, generateBoolNetTimeseries
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