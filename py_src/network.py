import itertools
import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np
from .general_utils import dissolve, check_probability_type_data, check_numeric
from .network_utils import convert_mined_result_to_fbn_network, remove_duplicates
from .network_app import filter_network_connections
import fbnnet_utils
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# def check_probability_type_data(value: float) -> None:
#     """Check if value is between 0 and 1"""
#     if not 0 <= value <= 1:
#         raise ValueError(f"Value {value} must be between 0 and 1")

# def check_numeric(value: Any) -> None:
#     """Check if value is numeric"""
#     if not isinstance(value, (int, float)):
#         raise ValueError(f"Value {value} must be numeric")

# def dissolve(nested_list: List) -> List:
#     """Flatten a nested list structure"""
#     result = []
#     for item in nested_list:
#         if isinstance(item, list):
#             result.extend(dissolve(item))
#         elif item is not None:
#             result.append(item)
#     return result



# def split_expression(expr: str, split_by: str = ",", strip: bool = True) -> List[str]:
#     """Split an expression string"""
#     if not expr:
#         return []
#     parts = expr.split(split_by)
#     if strip:
#         parts = [p.strip() for p in parts]
#     return parts

def mine_fbn_network(
    fbn_gene_cube: Dict,
    genes: Optional[List[str]] = None,
    use_parallel: bool = False,
    threshold_confidence: float = 1,
    threshold_error: float = 0,
    threshold_support: float = 1e-5,
    max_fbn_rules: int = 5
) -> Dict:
    """
    Mine FBN Networks from an Orchard cube
    
    Args:
        fbn_gene_cube: A pre constructed Orchard cube
        genes: The target genes in the output
        use_parallel: An option turns on parallel processing
        threshold_confidence: Threshold of confidence (between 0 and 1)
        threshold_error: Threshold of error rate (between 0 and 1)
        threshold_support: Threshold of support (between 0 and 1)
        max_fbn_rules: Maximum rules per type (Activation and Inhibition) per gene
    
    Returns:
        A FBN network object
    """
    logger.info(f"Enter mine_fbn_network zone: genes={genes}, use_parallel={use_parallel}")
    
    # Input validation
    check_probability_type_data(threshold_confidence)
    check_probability_type_data(threshold_error)
    check_probability_type_data(threshold_support)
    check_numeric(max_fbn_rules)
    
    if not fbn_gene_cube:
        return None
    
    if genes is None:
        genes = fbn_gene_cube['target_genes']

    genes = [gene for gene in genes if gene in fbn_gene_cube['cube'].keys()]

    middle_result = search_fbn_core(
        fbn_gene_cube['cube'], 
        genes, 
        use_parallel, 
        threshold_confidence, 
        threshold_error, 
        threshold_support, 
        max_fbn_rules
    )
    
    final_result = mine_fbn_network_with_cores(middle_result, genes, threshold_error, max_fbn_rules)
    logger.info("Leave mine_fbn_network zone")
    return final_result

def mine_fbn_network_with_cores(
    search_fbn_core_result: Dict,
    genes: Optional[List[str]] = None,
    threshold_error: float = 0,
    max_fbn_rules: int = 5
) -> Dict:
    """
    Process the result from search_fbn_core
    
    Args:
        search_fbn_core_result: Result from search_fbn_core
        genes: Genes that are involved
        threshold_error: Threshold of error rate
        max_fbn_rules: Maximum rules per type per gene
    
    Returns:
        A FBN network object
    """
    logger.info("Enter mine_fbn_network_with_cores zone")
    
    if genes is None:
        genes = list(search_fbn_core_result.keys())
    
    midle_result = mine_fbn_network_stage2(search_fbn_core_result, threshold_error, max_fbn_rules)
    final_result = convert_mined_result_to_fbn_network(midle_result, genes)
    
    if not final_result:
        raise ValueError("No Network generated")
    
    final_result = filter_network_connections(final_result)
    logger.info("Leave mine_fbn_network_with_cores zone")
    return final_result

def search_fbn_core(
    fbn_gene_cube: Dict,
    genes: List[str],
    use_parallel: bool = False,
    threshold_confidence: float = 1,
    threshold_error: float = 0,
    threshold_support: float = 1e-4,
    max_fbn_rules: int = 5
) -> Dict:
    """
    Core function to search for FBN rules in the cube
    
    Args:
        fbn_gene_cube: The orchard cube
        genes: Target genes
        use_parallel: Whether to use parallel processing
        threshold_confidence: Confidence threshold
        threshold_error: Error threshold
        threshold_support: Support threshold
        max_fbn_rules: Maximum rules per type per gene
    
    Returns:
        Dictionary of mined rules
    """
    if use_parallel:
        use_parallel = False
        logger.info("The parallel for network is not supported")
    
    logger.info(f"Enter search_fbn_core zone: use_parallel={use_parallel}")
    
    network_configs = {
        'threshold_confidence': threshold_confidence,
        'threshold_error': threshold_error,
        'threshold_support': threshold_support,
        'max_fbn_rules': max_fbn_rules
    }
    
    def recursive_mining_fbn_function(
        fbn_gene_cube_stem: Dict,
        target_gene: str,
        current_sub_gene: str,
        group_by_gene: Optional[str],
        configs: Dict,
        find_true: bool,
        find_false: bool
    ) -> List[Dict]:
        """Recursive function to mine FBN rules"""
        threshold_confidence = configs['threshold_confidence']
        threshold_error = configs['threshold_error']
        threshold_support = configs['threshold_support']
        
        if current_sub_gene not in fbn_gene_cube_stem:
            return None
        
        if group_by_gene is None:
            group_by_gene = current_sub_gene
        
        res = []
        res_t = []
        res_f = []
        
        # Get measures at current level
        input_genes = sorted(fbn_gene_cube_stem[current_sub_gene].get("Input", []))
        pick_t = fbn_gene_cube_stem[current_sub_gene].get("ActivatorAndInhibitor", {}).get("Activator", None)
        pick_f = fbn_gene_cube_stem[current_sub_gene].get("ActivatorAndInhibitor", {}).get("Inhibitor", None)
        
        # Process activator
        if pick_t is not None:
            pick_t_value = float(pick_t.get("Confidence", 0))
            pick_t_all_confidence = float(pick_t.get("all_confidence", 0))
            pick_t_support = float(pick_t.get("support", 0))
            pick_t_causality_test = float(pick_t.get("causality_test", 0))
            error_activator = float(pick_t.get("Noise", 0))
            identity_t = pick_t.get("Identity", "")
            pick_t_type = pick_t.get("type", "")
            timestep_t = int(pick_t.get("timestep", 1))
            best_fit_p = float(pick_t.get("bestFitP", 0))
        else:
            pick_t_value = 0
            pick_t_all_confidence = 0
            pick_t_support = 0
            pick_t_causality_test = 0
            error_activator = 0
            identity_t = ""
            pick_t_type = ""
            timestep_t = 1
            best_fit_p = 0
        
        # Process inhibitor
        if pick_f is not None:
            pick_f_value = float(pick_f.get("Confidence", 0))
            pick_f_all_confidence = float(pick_f.get("all_confidence", 0))
            pick_f_support = float(pick_f.get("support", 0))
            pick_f_causality_test = float(pick_f.get("causality_test", 0))
            error_inhibitor = float(pick_f.get("Noise", 0))
            identity_f = pick_f.get("Identity", "")
            pick_f_type = pick_f.get("type", "")
            timestep_f = int(pick_f.get("timestep", 1))
            best_fit_n = float(pick_f.get("bestFitN", 0))
        else:
            pick_f_value = 0
            pick_f_all_confidence = 0
            pick_f_support = 0
            pick_f_causality_test = 0
            error_inhibitor = 0
            identity_f = ""
            pick_f_type = ""
            timestep_f = 1
            best_fit_n = 0
        
        # Filter rules based on thresholds
        get_t = None
        if (pick_t is not None and 
            pick_t_support >= threshold_support and 
            pick_t_causality_test >= 1 and 
            pick_t_value >= threshold_confidence):
            get_t = pick_t
        
        get_f = None
        if (pick_f is not None and 
            pick_f_support >= threshold_support and 
            pick_f_causality_test >= 1 and 
            pick_f_value >= threshold_confidence):
            get_f = pick_f
        
        # Add rules to results if they pass thresholds
        if get_t is not None:
            res.append({
                'targets': target_gene,
                'factor': get_t['factor'],
                'type': 1,
                'identity': identity_t,
                'error': round(error_activator, 5),
                'P': round(pick_t_value, 4),
                'support': pick_t_support,
                'timestep': timestep_t,
                'input': ",".join(input_genes),
                'numOfInput': len(input_genes),
                'causality_test': round(pick_t_causality_test, 5),
                'GroupBy': group_by_gene,
                'all_confidence': pick_t_all_confidence,
                'dimensionType': pick_t_type,
                'bestFitP': best_fit_p
            })
        
        if get_f is not None:
            res.append({
                'targets': target_gene,
                'factor': get_f['factor'],
                'type': 0,
                'identity': identity_f,
                'error': round(error_inhibitor, 5),
                'P': round(pick_f_value, 4),
                'support': pick_f_support,
                'timestep': timestep_f,
                'input': ",".join(input_genes),
                'numOfInput': len(input_genes),
                'causality_test': round(pick_f_causality_test, 5),
                'GroupBy': group_by_gene,
                'all_confidence': pick_f_all_confidence,
                'dimensionType': pick_f_type,
                'bestFitN': best_fit_n
            })
        
        # Update find flags
        find_true = find_true or (get_t is not None)
        find_false = find_false or (get_f is not None)
        
        # Recursively process sub-levels if needed
        if not (find_true and find_false):
            # Process activator sub-genes
            if "SubGenesT" in fbn_gene_cube_stem[current_sub_gene]:
                fbn_gene_cube_stem_t = fbn_gene_cube_stem[current_sub_gene]["SubGenesT"]
                next_genes_t = list(fbn_gene_cube_stem_t.keys())
                
                for next_gene in next_genes_t:
                    sub_res = recursive_mining_fbn_function(
                        fbn_gene_cube_stem_t, target_gene, next_gene, 
                        group_by_gene, configs, find_true, find_false
                    )
                    if sub_res:
                        res_t.extend(dissolve(sub_res))
            
            # Process inhibitor sub-genes
            if "SubGenesF" in fbn_gene_cube_stem[current_sub_gene]:
                fbn_gene_cube_stem_f = fbn_gene_cube_stem[current_sub_gene]["SubGenesF"]
                next_genes_f = list(fbn_gene_cube_stem_f.keys())
                
                for next_gene in next_genes_f:
                    sub_res = recursive_mining_fbn_function(
                        fbn_gene_cube_stem_f, target_gene, next_gene, 
                        group_by_gene, configs, find_true, find_false
                    )
                    if sub_res:
                        res_f.extend(dissolve(sub_res))
        
        # Combine results
        if res_t:
            res.extend(res_t)
        if res_f:
            res.extend(res_f)
        
        # Filter out None and empty results
        res = [r for r in res if r is not None and len(r) > 0]
        return res
    
    def internal_loop(i: int, genes: List[str], env: Dict) -> List[Dict]:
        """Internal processing loop for each gene"""
        target_gene = genes[i]
        res = {target_gene: []}
        
        if not env['cube'] or target_gene not in env['cube']:
            return []
        
        if "SubGenes" not in env['cube'][target_gene]:
            return []
        
        current_stem = env['cube'][target_gene]["SubGenes"]
        next_genes = list(current_stem.keys())
        
        for current_gene in next_genes:
            sub_res = recursive_mining_fbn_function(
                current_stem, target_gene, current_gene, 
                None, env['configs'], False, False
            )
            if sub_res:
                res[target_gene].extend(dissolve(sub_res))
        
        # Remove duplicates
        preresponse = remove_duplicates(dissolve(res[target_gene]))
        if not preresponse:
            return []
        
        return {target_gene: preresponse}
    
    # Main processing
    if not genes or not fbn_gene_cube:
        return {}
    
    main_parameters = {
        'cube': fbn_gene_cube,
        'configs': network_configs
    }
    
    # Process each gene
    results = []
    for i in range(len(genes)):
        gene_result = internal_loop(i, genes, main_parameters)
        if gene_result:
            results.append(gene_result)
    
    # Combine results
    final_result = {}
    for r in results:
        final_result.update(r)
    
    # Filter out empty results
    final_result = {k: v for k, v in final_result.items() if v}
    
    logger.info("Leave search_fbn_core zone")
    return final_result

def _drop_subsumed_rules(ruleset: List[Dict]) -> List[Dict]:
    """Drop a rule when another rule of the same type/timestep has fewer inputs, all contained in it.

    R's mineFBNNetworkStage2 compares every rule against every other one; this gives the same
    result by tokenising once and looking up each input-gene subset of a rule in a dict.
    """
    tokens = [frozenset(fbnnet_utils.splitExpression(r['input'], 2, False)) for r in ruleset]
    n_inputs = [int(r['numOfInput']) for r in ruleset]
    groups = [(int(r['type']), int(r['timestep'])) for r in ruleset]

    fewest_inputs: Dict[Any, int] = {}
    for group, tok, n in zip(groups, tokens, n_inputs):
        key = (group, tok)
        if n < fewest_inputs.get(key, n + 1):
            fewest_inputs[key] = n

    kept = []
    for i, rule in enumerate(ruleset):
        tok, n, group = tokens[i], n_inputs[i], groups[i]
        if len(tok) <= 12:
            subsumed = any(
                fewest_inputs.get((group, frozenset(sub)), n) < n
                for size in range(len(tok) + 1)
                for sub in itertools.combinations(tok, size)
            )
        else:
            subsumed = any(
                j != i and groups[j] == group and n_inputs[j] < n and tokens[j] <= tok
                for j in range(len(ruleset))
            )
        if not subsumed:
            kept.append(rule)
    return kept


def mine_fbn_network_stage2(
    res: Dict,
    threshold_error: float = 0,
    max_fbn_rules: int = 5
) -> Dict:
    """
    Second stage of FBN network mining
    
    Args:
        res: Result from stage 1
        threshold_error: Error threshold
        max_fbn_rules: Maximum rules per type per gene
    
    Returns:
        Filtered FBN rules
    """
    logger.info("Enter mine_fbn_network_stage2 zone")
    
    if not res:
        return {}
    
    filtered_res = {}
    
    for target, ruleset in res.items():
        filtered_res[target] = _drop_subsumed_rules(ruleset)
    
    # Filter rules based on error threshold and type
    for target in filtered_res:
        activators = []
        inhibitors = []
        
        for rule in filtered_res[target]:
            if (float(rule['error']) <= threshold_error and 
                int(rule['type']) == 1):
                rule['threshold_error'] = threshold_error
                activators.append(rule)
            
            if (float(rule['error']) <= threshold_error and 
                int(rule['type']) == 0):
                rule['threshold_error'] = threshold_error
                inhibitors.append(rule)
        
        # Sort and limit number of rules
        activators.sort(key=lambda x: (
            int(x['timestep']), 
            float(x['error']), 
            int(x['numOfInput']), 
            -float(x['support'])
        ))
        if len(activators) > max_fbn_rules:
            activators = activators[:max_fbn_rules]
        
        inhibitors.sort(key=lambda x: (
            int(x['timestep']), 
            float(x['error']), 
            int(x['numOfInput']), 
            -float(x['support'])
        ))
        if len(inhibitors) > max_fbn_rules:
            inhibitors = inhibitors[:max_fbn_rules]
        
        filtered_res[target] = activators + inhibitors
    
    # Filter out empty entries
    filtered_res = {k: v for k, v in filtered_res.items() if v}
    
    logger.info("Leave mine_fbn_network_stage2 zone")
    return filtered_res

# # Helper functions that would need to be implemented
# def convert_mined_result_to_fbn_network(mined_result: Dict, genes: List[str]) -> Dict:
#     """Convert mined result to FBN network format"""
#     # Implementation would depend on your specific network format requirements
#     return mined_result

# def filter_network_connections(network: Dict) -> Dict:
#     """Filter network connections"""
#     # Implementation would depend on your specific filtering requirements
#     return network

# Example usage (assuming we have the required input data)
if __name__ == "__main__":
    # This would be replaced with actual data loading
    example_network = {
        'genes': ['Gene1', 'Gene2', 'Gene3'],
        # ... other network properties
    }
    
    # Generate initial states and time series (placeholder)
    initial_states = []  # generate_all_combination_binary(example_network['genes'])
    training_series = []  # generate_boolnet_timeseries(example_network, initial_states, 43, type='synchronous')
    
    # Construct FBN cube (placeholder)
    cube = {}  # construct_fbn_cube(example_network['genes'], example_network['genes'], training_series, maxK=4, temporal=1, useParallel=False)
    
    # Mine network
    network = mine_fbn_network(cube, example_network['genes'])
    print(network)