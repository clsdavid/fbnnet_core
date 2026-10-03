import multiprocessing
import os
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import logging
from . import _tree
from . import _core
from . import _matrix
from .general_utils import fbn_data_reduction


# Set up logging
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
    process_fn = _process_cube_algorithm_for_backend(state.get("backend", "recursive"))
    return process_fn(
        gene, state["conditional_genes"], state["max_k"], state["temporal"],
        state["main_parameters"], None, None,
    )


def _process_cube_algorithm_for_backend(backend: str):
    """Returns the process_cube_algorithm callable for the requested backend.

    'recursive' (default) is the original, ground-truth C++ engine
    (_tree.process_cube_algorithm). 'tensor' is the Phase 3 node-level
    candidate-gene batching backend (fbnnet_core/tensor_mining.py) - same
    recursion/branching control flow, but the per-node "try every candidate
    gene" loop is vectorized with NumPy. Validated byte-identical against
    the recursive backend (tests/test_tensor_mining.py).
    """
    if backend == "recursive":
        return _tree.process_cube_algorithm
    if backend == "tensor":
        from .tensor_mining import process_cube_algorithm as tensor_process_cube_algorithm
        return tensor_process_cube_algorithm
    raise ValueError(f"Unknown backend '{backend}'; expected 'recursive' or 'tensor'")

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
        reduced_cube[i] = _matrix.FBNMatrix(mat, genes_input, [str(j+1) for j in range(mat.shape[1])])
    # Initialize state containers
    current_states = [None] * temporal
    previous_states = [None] * temporal
    current_states_c = [None] * temporal
    previous_states_c = [None] * temporal
    
    # Set up data for each temporal level
    for index in range(temporal, 0, -1):
        current_states[index-1] = _core.extract_gene_state_from_time_series_cube(reduced_cube, index)
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
                      max_workers: Optional[int] = None,
                      chunksize: Optional[int] = None,
                      backend: str = "recursive") -> Dict[str, Any]:
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
    chunksize : int, optional
        Tasks-per-worker batch size passed to the underlying pool.map when
        use_parallel is True (default: pool.map's own heuristic). Tune this
        upward for very large gene counts to reduce IPC/dispatch overhead.
    backend : str, optional
        'recursive' (default) uses the original compiled C++ mining engine
        (_tree.process_cube_algorithm). 'tensor' uses the Phase 3
        node-level candidate-gene batching backend (fbnnet_core/tensor_mining.py),
        a NumPy reimplementation validated byte-identical to 'recursive' but
        vectorized across candidate genes at each tree node.
    
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
    #     reduced_cube[i] = _matrix.FBNMatrix(mat, genes_input, [str(j+1) for j in range(mat.shape[1])])
    # # Initialize state containers
    # current_states = [None] * temporal
    # previous_states = [None] * temporal
    # current_states_c = [None] * temporal
    # previous_states_c = [None] * temporal
    
    # # Set up data for each temporal level
    # for index in range(temporal, 0, -1):
    #     current_states[index-1] = _core.extract_gene_state_from_time_series_cube(reduced_cube, index)
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
    if backend not in ("recursive", "tensor"):
        raise ValueError(f"Unknown backend '{backend}'; expected 'recursive' or 'tensor'")

    main_parameters = convert_df_main_parameters(timeseries_cube, temporal)
    target_genes = _tree.filterTargetGenesByConditionGenes(target_genes, main_parameters, conditional_genes, None, temporal)
    # Process each target gene
    if use_parallel:
        res = do_parallel_work(target_genes, conditional_genes, max_k, temporal, main_parameters, max_workers, chunksize, backend)
    else:
        res = do_non_parallel_work(target_genes, conditional_genes, max_k, temporal, main_parameters, backend)
    
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
                     max_workers: Optional[int] = None,
                     chunksize: Optional[int] = None,
                     backend: str = "recursive") -> List[Dict[str, Any]]:
    """Process genes in parallel using separate OS processes (fork), not threads.

    The pybind11-based main_parameters object is never pickled: it's stashed in a
    module-level global before the fork-context Pool is created, so forked workers
    inherit it via copy-on-write memory. Only the per-task gene name (a plain
    string) is pickled through the task queue. This replaces an earlier
    ThreadPoolExecutor-based implementation that provided no real speedup because
    none of the pybind11 bindings release the GIL.

    chunksize is forwarded to pool.map verbatim; leave it None to use the
    pool's own heuristic (fine for the common case), and set it explicitly
    when mining very large gene counts if profiling shows per-task dispatch
    overhead is non-trivial relative to per-gene work.
    """
    global _cube_worker_state

    n_workers = max(1, min(max_workers or DEFAULT_MAX_WORKERS, len(target_genes)))
    if not _FORK_AVAILABLE or n_workers <= 1:
        if not _FORK_AVAILABLE:
            logger.warning(
                "multiprocessing 'fork' start method unavailable on this platform; "
                "falling back to sequential execution."
            )
        return do_non_parallel_work(target_genes, conditional_genes, max_k, temporal, main_parameters, backend)

    _cube_worker_state = {
        "conditional_genes": conditional_genes,
        "max_k": max_k,
        "temporal": temporal,
        "main_parameters": main_parameters,
        "backend": backend,
    }
    try:
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(processes=n_workers) as pool:
            results = pool.map(_cube_worker, target_genes, chunksize=chunksize)
    finally:
        _cube_worker_state = {}

    return results

def do_non_parallel_work(target_genes: List[str], 
                        conditional_genes: List[str], 
                        max_k: int, 
                        temporal: int, 
                        main_parameters: Dict[str, Any],
                        backend: str = "recursive") -> List[Dict[str, Any]]:
    """Process genes sequentially"""
    process_fn = _process_cube_algorithm_for_backend(backend)
    results = []
    for gene in target_genes:
        results.append(
            process_fn(gene, conditional_genes, max_k, temporal, main_parameters, None, None)
        )
    return results
