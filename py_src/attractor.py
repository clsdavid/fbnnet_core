"""
Port of R/attractor_FBN.R and the attractor-related parts of R/modelling_FBN.R:
searchForAttractors, getFBMSuccessor, isSatisfied, randomSelection,
getProbabilityFromFunctionInput and networkFixUpdate.
"""
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import fbnnet_utils

from .data_utils import generateAllCombinationBinary
from .network_app import _as_gene_value_dict


def is_satisfied(gene_state: Dict[str, int], expression_tokens: List[str]) -> bool:
    """
    Check whether a gene state (input gene name -> 0/1 value) satisfies a
    tokenized Boolean expression (as produced by splitExpression(expr, 1, False)).
    """
    if expression_tokens == ["1"]:
        return True
    if expression_tokens == ["0"]:
        return False

    res = True
    for name, value in gene_state.items():
        if name not in expression_tokens:
            # The gene isn't part of this expression; nothing to check.
            continue
        index = expression_tokens.index(name)
        if index > 0 and expression_tokens[index - 1] == "!":
            res = res and not (int(value) == 1)
        else:
            res = res and (int(value) == 1)
    return res


def random_selection(probability: float) -> bool:
    """A random selection function denoted as P[] in the R source."""
    return random.random() < probability


def get_probability_from_function_input(
    func_type: int,
    expression: str,
    probability: float,
    pre_gene_inputs: Dict[str, int],
) -> float:
    """Get the probability of a regulatory function given the current inputs."""
    if func_type not in (0, 1):
        raise ValueError('The function type value must be in the range of "0" and "1"')

    tokens = fbnnet_utils.splitExpression(expression, 1, False)
    if is_satisfied(pre_gene_inputs, tokens):
        return float(probability)
    return 0.0


def _internal_fun(
    gene: str,
    interactions: Dict,
    gene_state: bool,
    previous_states: np.ndarray,
    current_step: int,
    genes: List[str],
    timedecay: float,
    decay_index: int = 1,
) -> Tuple[bool, int]:
    """Compute the next value of a single gene and its updated decay index."""
    gene_functions = interactions.get(gene, {})
    items = list(gene_functions.values()) if isinstance(gene_functions, dict) else list(gene_functions)

    ini = gene_state
    decay = timedecay if timedecay and timedecay >= 1 else 1

    func_of_activators = [i for i in items if int(i['type']) == 1]
    func_of_inhibitors = [i for i in items if int(i['type']) == 0]

    timestep_applied = False

    def _pre_gene_input(interaction: Dict, adapted_timestep: int) -> Dict[str, int]:
        return {
            genes[idx - 1]: int(previous_states[idx - 1, adapted_timestep - 1])
            for idx in interaction['input']
        }

    pr_fa = False
    for interaction in func_of_activators:
        adapted_timestep = current_step - int(interaction.get('timestep', 1))
        if previous_states.shape[1] < adapted_timestep or adapted_timestep < 1:
            timestep_applied = True
            continue
        pre_gene_input = _pre_gene_input(interaction, adapted_timestep)
        probability = get_probability_from_function_input(
            1, interaction['expression'], interaction['probability'], pre_gene_input
        )
        if probability:
            pr_fa = pr_fa or random_selection(probability)

    pr_fd = False
    for interaction in func_of_inhibitors:
        adapted_timestep = current_step - int(interaction.get('timestep', 1))
        if previous_states.shape[1] < adapted_timestep or adapted_timestep < 1:
            timestep_applied = True
            continue
        pre_gene_input = _pre_gene_input(interaction, adapted_timestep)
        probability = get_probability_from_function_input(
            0, interaction['expression'], interaction['probability'], pre_gene_input
        )
        if probability:
            pr_fd = pr_fd or random_selection(probability)

    if decay > 0 and not timestep_applied:
        if pr_fa or pr_fd:
            decay_index = 1
        else:
            if decay_index >= decay:
                ini = False
                decay_index = 1
            else:
                decay_index = decay_index + 1

    result = (ini or pr_fa) and (not pr_fd)
    return result, decay_index


def get_fbm_successor(
    fbn_network: Dict,
    previous_states: np.ndarray,
    current_step: int,
    genes: List[str],
    transition_type: str = "synchronous",
    decay_index: Optional[Dict[str, int]] = None,
) -> Dict:
    """
    Compute the next state of a Fundamental Boolean Network given its history.

    Args:
        fbn_network: A FundamentalBooleanNetwork (interactions/genes/fixed/timedecay)
        previous_states: A genes x timesteps array of prior 0/1 states
        current_step: The 1-based index of the step being computed
        genes: Ordered gene names matching previous_states rows and interaction inputs
        transition_type: "synchronous" or "asynchronous"
        decay_index: Optional per-gene decay counters (gene -> int)

    Returns:
        {'nextState': {gene: 0/1}, 'decayIndex': {gene: int}}
    """
    if transition_type not in ("synchronous", "asynchronous"):
        raise ValueError('type must be "synchronous" or "asynchronous"')

    interactions = fbn_network['interactions']
    fixed = _as_gene_value_dict(fbn_network.get('fixed', {}), genes)
    timedecay = _as_gene_value_dict(fbn_network.get('timedecay', {}), genes, default=1)

    fixed_genes = {g for g in genes if fixed.get(g, -1) == 0}

    if decay_index is None:
        decay_index = {gene: 1 for gene in genes}
    new_decay_index = dict(decay_index)

    gene_index = {gene: i for i, gene in enumerate(genes)}
    prior_col = previous_states.shape[1] - 1

    next_state = {gene: int(previous_states[gene_index[gene], prior_col]) for gene in genes}

    if transition_type == "asynchronous":
        non_fixed_genes = [g for g in genes if g not in fixed_genes]
        gene = random.choice(non_fixed_genes)
        ini = previous_states[gene_index[gene], prior_col] == 1
        result, new_decay = _internal_fun(
            gene, interactions, ini, previous_states, current_step, genes,
            timedecay.get(gene, -1), decay_index.get(gene, 1),
        )
        new_decay_index[gene] = new_decay
        next_state[gene] = 1 if result else 0
    else:
        for gene in genes:
            if gene in fixed_genes:
                new_decay_index[gene] = 0
                continue
            ini = previous_states[gene_index[gene], prior_col] == 1
            result, new_decay = _internal_fun(
                gene, interactions, ini, previous_states, current_step, genes,
                timedecay.get(gene, -1), decay_index.get(gene, 1),
            )
            next_state[gene] = 1 if result else 0
            new_decay_index[gene] = new_decay

    return {'nextState': next_state, 'decayIndex': new_decay_index}


def transition_states(
    initial_state: Dict[str, int],
    fbn_network: Dict,
    genes: List[str],
    transition_type: str = "synchronous",
    max_timepoints: int = 100,
) -> np.ndarray:
    """
    Port of R's `transitionStates` (modelling_FBN.R).

    Simulates a single trajectory of `max_timepoints` steps starting from
    `initial_state`, using get_fbm_successor at each step.

    Returns:
        A genes x max_timepoints ndarray of 0/1 states.
    """
    mat = np.zeros((len(genes), max_timepoints), dtype=int)
    mat[:, 0] = [initial_state[gene] for gene in genes]
    decay_index = {gene: 1 for gene in genes}
    for k in range(2, max_timepoints + 1):
        result = get_fbm_successor(
            fbn_network, mat[:, :k - 1], k, genes, transition_type, decay_index
        )
        mat[:, k - 1] = [result['nextState'][gene] for gene in genes]
        decay_index = result['decayIndex']
    return mat


def reconstruct_timeseries(
    fbn_network: Dict,
    initial_states: List[Dict[str, int]],
    transition_type: str = "synchronous",
    max_timepoints: int = 100,
    use_parallel: bool = False,
) -> List[np.ndarray]:
    """
    Port of R's `reconstructTimeseries` (modelling_FBN.R).

    Reconstructs a time series matrix (genes x max_timepoints) for each state
    in `initial_states` by simulating transitions through `fbn_network`.

    Args:
        fbn_network: A FundamentalBooleanNetwork
        initial_states: A list of gene-name -> 0/1 dicts, one per starting state
        transition_type: "synchronous" or "asynchronous"
        max_timepoints: Number of timepoints to simulate per trajectory
        use_parallel: Unused; kept for signature parity with the R source.

    Returns:
        A list of genes x max_timepoints ndarrays, one per initial state.
    """
    if fbn_network.get('class') != 'FundamentalBooleanNetwork':
        raise ValueError("Network must be inherited from FundamentalBooleanNetwork")
    if not isinstance(max_timepoints, int) or max_timepoints <= 0:
        raise ValueError("maxTimepoints must be a positive integer")

    genes = fbn_network['genes']
    return [
        transition_states(state, fbn_network, genes, transition_type, max_timepoints)
        for state in initial_states
    ]


def network_fix_update(network: Dict, fix_genes: List[str], values: List[int]) -> Dict:
    """
    Fix specific genes in a FundamentalBooleanNetwork.

    Args:
        network: A FundamentalBooleanNetwork
        fix_genes: The genes to fix
        values: 0 (fix off), 1 (fix on) or -1 (unfix) per gene (or a single value for all)

    Returns:
        The network with an updated 'fixed' dict
    """
    if network.get('class') != 'FundamentalBooleanNetwork':
        raise ValueError("Network must be inherited from FundamentalBooleanNetwork")

    if len(values) == 1:
        values = values * len(fix_genes)
    if len(fix_genes) != len(values):
        raise ValueError("fixIndices and values must have the same number of elements!")

    if any(gene not in network['genes'] for gene in fix_genes):
        raise ValueError("fixIndices contains invalid indices!")

    if any(v not in (0, 1, -1) for v in values):
        raise ValueError("Please supply only 0, 1, or -1 in values!")

    fixed = dict(network.get('fixed', {}))
    for gene, value in zip(fix_genes, values):
        fixed[gene] = int(value)
    network['fixed'] = fixed
    return network


def search_for_attractors(
    fbn_network: Dict,
    genes: List[str],
    start_states: Optional[List[Dict[str, int]]] = None,
    transition_type: str = "synchronous",
    genes_on: Optional[List[str]] = None,
    genes_off: Optional[List[str]] = None,
    max_search: int = 1000,
) -> Dict:
    """
    Find all possible Fundamental Boolean Model attractors.

    Args:
        fbn_network: A FundamentalBooleanNetwork
        genes: Ordered gene names matching the network
        start_states: Optional list of initial states (gene -> 0/1 dicts);
            defaults to all binary combinations of genes
        transition_type: "synchronous" or "asynchronous"
        genes_on: Genes to force to 1 in every start state
        genes_off: Genes to force to 0 in every start state
        max_search: Maximum number of timesteps to search per start state

    Returns:
        {'Attractors': [...], 'Genes': genes, 'BasinOfAttractor': [...], 'class': 'FBMAttractors'}
    """
    if fbn_network.get('class') != 'FundamentalBooleanNetwork':
        raise ValueError("Network must be inherited from FundamentalBooleanNetwork")
    if transition_type not in ("synchronous", "asynchronous"):
        raise ValueError('type must be "synchronous" or "asynchronous"')

    if not start_states:
        start_states = generateAllCombinationBinary(genes)

    genes_on = genes_on or []
    genes_off = genes_off or []
    start_states = [dict(state) for state in start_states]
    for state in start_states:
        for gene in genes_on:
            state[gene] = 1
        for gene in genes_off:
            state[gene] = 0

    result_list: List[List[Dict[str, int]]] = []
    basin_states: List[List[Dict[str, int]]] = []
    state_assigned: List[Dict[str, int]] = []

    for start_state in start_states:
        if start_state in state_assigned:
            continue

        searched_states = [dict(start_state)]
        temp_basin_states = [dict(start_state)]

        mat = np.zeros((len(genes), max_search), dtype=int)
        mat[:, 0] = [start_state[gene] for gene in genes]

        decay_index = {gene: 1 for gene in genes}
        found = False
        max_s = 1

        while not found and max_s <= max_search:
            k = max_s + 1
            result = get_fbm_successor(fbn_network, mat[:, :k - 1], k, genes, transition_type, decay_index)
            next_state = result['nextState']
            decay_index = result['decayIndex']

            if next_state in state_assigned:
                break

            found_basin_index = None
            for idx, states in enumerate(basin_states):
                if next_state in states:
                    found_basin_index = idx
                    break
            if found_basin_index is not None:
                existing = basin_states[found_basin_index]
                for s in temp_basin_states:
                    if s not in existing:
                        existing.append(s)
                break

            if next_state in searched_states:
                found = True
                existing_attractors = [attractor[0] for attractor in result_list]
                if next_state not in existing_attractors:
                    cycle_start = searched_states.index(next_state)
                    attractor_cycle = searched_states[cycle_start:] + [next_state]
                    result_list.append(attractor_cycle)
                    basin = [s for s in temp_basin_states if s not in attractor_cycle]
                    basin_states.append(basin)
                    for s in attractor_cycle:
                        if s not in state_assigned:
                            state_assigned.append(s)
            else:
                searched_states.append(next_state)
                temp_basin_states.append(next_state)

            mat[:, k - 1] = [next_state[gene] for gene in genes]
            max_s += 1

    return {
        'Attractors': result_list,
        'Genes': genes,
        'BasinOfAttractor': basin_states,
        'class': 'FBMAttractors',
    }
