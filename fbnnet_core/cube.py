import os
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import logging
from . import _parallel
from . import _tree
from . import _core
from . import _matrix
from .general_utils import fbn_data_reduction


# Set up logging
logger = logging.getLogger(__name__)

# "cores - 1" convention: leave one core free for the OS/orchestrator process.
DEFAULT_MAX_WORKERS = max(1, (os.cpu_count() or 2) - 1)

# Populated in the parent process right before a fork-context Pool is created.
# Forked workers inherit this via copy-on-write memory, so the (potentially
# unpicklable) pybind11 main_parameters object is never pickled across the
# process boundary -- only the per-task gene name is.
_cube_worker_state: Dict[str, Any] = {}


def _cube_worker_init(main_parameters, timeseries_cube, conditional_genes, max_k, temporal, backend,
                      postprocess=None, check_essential=False) -> None:
    """Sets the worker state; under spawn (main_parameters is None) rebuilds it from the data."""
    global _cube_worker_state
    if main_parameters is None:
        main_parameters = convert_df_main_parameters(timeseries_cube, temporal)
        if backend == "tensor":
            from . import tensor_nodes

            tensor_nodes.prepare(main_parameters, temporal)
    _cube_worker_state = {
        "conditional_genes": conditional_genes,
        "max_k": max_k,
        "temporal": temporal,
        "main_parameters": main_parameters,
        "backend": backend,
        "postprocess": postprocess,
        "check_essential": check_essential,
    }


def _cube_worker_clear() -> None:
    global _cube_worker_state
    _cube_worker_state = {}


def _mine_target(gene, conditional_genes, max_k, temporal, main_parameters, backend, postprocess, check_essential):
    """The cube entry of one target (None when `check_essential` finds no essential candidate gene),
    passed through `postprocess` when given."""
    result = None
    skip = False
    if check_essential:
        from . import tensor_nodes

        cache = tensor_nodes.prepare(main_parameters, temporal)
        skip = not tensor_nodes.has_essential_candidate(cache, gene, conditional_genes, temporal)
    if not skip:
        process_fn = _process_cube_algorithm_for_backend(backend)
        result = process_fn(gene, conditional_genes, max_k, temporal, main_parameters, None, None)
    return postprocess(result) if postprocess is not None else result


def _cube_worker(gene: str) -> Any:
    """Runs in a worker process; reads the state set by `_cube_worker_init`."""
    state = _cube_worker_state
    return _mine_target(
        gene, state["conditional_genes"], state["max_k"], state["temporal"], state["main_parameters"],
        state.get("backend", "auto"), state.get("postprocess"), state.get("check_essential", False),
    )


BACKENDS = ("auto", "recursive", "tensor")


def resolve_backend(backend: str) -> str:
    """Maps 'auto' to a concrete backend and rejects unknown names.

    The tensor backend was never slower than the recursive one in the benchmarks (5 to 54,613
    genes, tools/benchmark_cube.py) and is 6-14x faster beyond a few hundred genes, so 'auto'
    selects it. This is the single place to refine the choice.
    """
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend '{backend}'; expected one of {', '.join(repr(b) for b in BACKENDS)}")
    return "tensor" if backend == "auto" else backend


def _process_cube_algorithm_for_backend(backend: str):
    """Returns the process_cube_algorithm callable for the requested backend.

    'recursive' is the original, ground-truth C++ engine
    (_tree.process_cube_algorithm). 'tensor' is the Phase 3 node-level
    candidate-gene batching backend (fbnnet_core/tensor_mining.py) - same
    recursion/branching control flow, but the per-node "try every candidate
    gene" loop is vectorized with NumPy. Validated byte-identical against
    the recursive backend (tests/test_tensor_mining.py).
    """
    backend = resolve_backend(backend)
    if backend == "recursive":
        return _tree.process_cube_algorithm
    from .tensor_mining import process_cube_algorithm as tensor_process_cube_algorithm
    return tensor_process_cube_algorithm

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
                      backend: str = "auto") -> Dict[str, Any]:
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
        'auto' (default) selects 'tensor'. 'recursive' uses the original compiled
        C++ mining engine (_tree.process_cube_algorithm). 'tensor' uses the Phase 3
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
    target_genes, res = run_cube_targets(
        target_genes, conditional_genes, timeseries_cube, max_k, temporal,
        use_parallel, max_workers, chunksize, backend,
    )
    
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



def run_cube_targets(target_genes: List[str],
                     conditional_genes: List[str],
                     timeseries_cube: List[pd.DataFrame],
                     max_k: int,
                     temporal: int,
                     use_parallel: bool = False,
                     max_workers: Optional[int] = None,
                     chunksize: Optional[int] = None,
                     backend: str = "auto",
                     postprocess=None,
                     filter_in_workers: bool = False) -> "tuple[List[str], List[Any]]":
    """Mines the Orchard cube entry of every target gene; returns (the targets kept, one result per target).

    Without `postprocess` each result is the cube entry ``{gene: {...}}`` (as in construct_fbn_cube).
    With it, `postprocess(entry)` runs in the worker right after the entry is built and its return value
    replaces the entry, so large entries never cross the process boundary. `postprocess` must be picklable
    (a module-level function or a `functools.partial` of one) where the start method is spawn.

    `filter_in_workers` (tensor backend, all genes present in the data) replaces the serial pre-filter of
    the targets, which is quadratic in the gene count, by the same test inside the workers; targets that
    fail it get None as their result (passed to `postprocess`), and the returned target list is then not
    filtered.
    """
    backend = resolve_backend(backend)

    main_parameters = convert_df_main_parameters(timeseries_cube, temporal)
    check_essential = False
    if filter_in_workers and backend == "tensor":
        known = set(main_parameters["rownames"])
        check_essential = all(g in known for g in target_genes) and all(g in known for g in conditional_genes)
    if not check_essential:
        target_genes = _tree.filterTargetGenesByConditionGenes(
            target_genes, main_parameters, conditional_genes, None, temporal
        )
    if use_parallel:
        res = do_parallel_work(
            target_genes, conditional_genes, max_k, temporal, main_parameters, max_workers, chunksize, backend,
            timeseries_cube, postprocess, check_essential,
        )
    else:
        res = do_non_parallel_work(
            target_genes, conditional_genes, max_k, temporal, main_parameters, backend, postprocess,
            check_essential,
        )
    return target_genes, res


def do_parallel_work(target_genes: List[str], 
                     conditional_genes: List[str], 
                     max_k: int, 
                     temporal: int, 
                     main_parameters: Dict[str, Any],
                     max_workers: Optional[int] = None,
                     chunksize: Optional[int] = None,
                     backend: str = "auto",
                     timeseries_cube: Optional[List[pd.DataFrame]] = None,
                     postprocess=None,
                     check_essential: bool = False) -> List[Dict[str, Any]]:
    """Process genes in parallel using separate OS processes, not threads.

    The pybind11-based main_parameters object is never pickled. With the fork start method it is
    stashed in a module-level global before the Pool is created, so workers inherit it via
    copy-on-write memory. With spawn (Windows) every worker rebuilds it from
    `timeseries_cube`, which is therefore required there. Only the per-task gene name (a plain
    string) is pickled through the task queue. This replaces an earlier
    ThreadPoolExecutor-based implementation that provided no real speedup because
    none of the pybind11 bindings release the GIL.

    chunksize is forwarded to pool.map verbatim; leave it None to use the
    pool's own heuristic (fine for the common case), and set it explicitly
    when mining very large gene counts if profiling shows per-task dispatch
    overhead is non-trivial relative to per-gene work.
    """
    backend = resolve_backend(backend)
    method = _parallel.start_method()
    n_workers = max(1, min(max_workers or DEFAULT_MAX_WORKERS, len(target_genes)))
    unusable = method is None or (method == "spawn" and timeseries_cube is None)
    if unusable and n_workers > 1:
        logger.warning(
            "No usable multiprocessing start method (spawn also needs the time series to rebuild "
            "the state in the workers); falling back to sequential execution."
        )
    if unusable or n_workers <= 1:
        return do_non_parallel_work(
            target_genes, conditional_genes, max_k, temporal, main_parameters, backend, postprocess,
            check_essential,
        )

    if method == "fork" and backend == "tensor":
        # Build the shared indicator matrices once, so that forked workers inherit them copy-on-write.
        from . import tensor_nodes

        tensor_nodes.prepare(main_parameters, temporal)

    return _parallel.pool_map(
        _cube_worker, target_genes, n_workers, chunksize,
        _cube_worker_init,
        (main_parameters, None, conditional_genes, max_k, temporal, backend, postprocess, check_essential),
        (None, timeseries_cube, conditional_genes, max_k, temporal, backend, postprocess, check_essential),
        _cube_worker_clear,
    )

def do_non_parallel_work(target_genes: List[str], 
                        conditional_genes: List[str], 
                        max_k: int, 
                        temporal: int, 
                        main_parameters: Dict[str, Any],
                        backend: str = "auto",
                        postprocess=None,
                        check_essential: bool = False) -> List[Dict[str, Any]]:
    """Process genes sequentially"""
    _process_cube_algorithm_for_backend(backend)  # rejects unknown backends before any work
    return [
        _mine_target(gene, conditional_genes, max_k, temporal, main_parameters, backend, postprocess, check_essential)
        for gene in target_genes
    ]
