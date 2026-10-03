import re
from typing import List, Dict, Optional, Union
from . import _utils
from .fbn_types import FundamentalBooleanNetwork
# Set up logging
import logging
logger = logging.getLogger(__name__)

def match_names(rule: str) -> List[str]:
        """Extract gene names from a rule expression"""
        regex = r"([_a-zA-Z][_a-zA-Z0-9]*)[,| |\)|\||\&|\[|\]]"
        rule = re.sub(r"\s+", "", rule) + " "
        # finditer + group(1) mirrors R's gregexpr+regexec (full match, then extract
        # the captured name) - re.findall would instead return the bare group text,
        # which can't be re-matched against a pattern that requires a trailing separator.
        genes = list(dict.fromkeys(m.group(1) for m in re.finditer(regex, rule)))
        
        # Remove operators
        operators = {"all", "any", "sumis", "sumgt", "sumlt", "maj", "timegt", "timelt", "timeis"}
        return [g for g in genes if g.lower() not in operators]

def load_fbn_network(file: str, body_separator: str = ",", lowercase_genes: bool = False) -> Dict:
    """
    Load FBN network data into a FBN network object
    
    Args:
        file: A file that contains FBN data
        body_separator: A specific text separator
        lowercase_genes: If all genes should be converted to lowercase
        
    Returns:
        An object of FBN network
    """

    
    try:
        with open(file, 'r') as f:
            func = f.readlines()
    except IOError as e:
        raise IOError(f"Error reading file: {e}")
    
    # Process lines
    func = [re.sub(r"#.*", "", line.strip()) for line in func]
    func = [line for line in func if line]
    
    if not func:
        raise ValueError("Header expected!")
    
    # Process header
    header = func[0]
    header = [h.strip().lower() for h in header.split(body_separator)]
    
    if len(header) < 3 or header[0] != "targets" or header[2] != "type":
        raise ValueError(f"Invalid header: {func[0]}")
    
    func = func[1:]
    
    if lowercase_genes:
        func = [line.lower() for line in func]
    
    # Clean up special characters
    func = [re.sub(r"[^\[\]a-zA-Z0-9_\|&!() \t\-+=.,]+", "_", line) for line in func]
    
    # Parse each line
    tmp = []
    for line in func:
        bracket_count = 0
        last_idx = 0
        res = []
        chars = list(line)
        
        for i, char in enumerate(chars):
            if char == "(":
                bracket_count += 1
            elif char == ")":
                bracket_count -= 1
            elif char == body_separator and bracket_count == 0:
                res.append("".join(chars[last_idx:i]).strip())
                last_idx = i + 1
        
        res.append("".join(chars[last_idx:]).strip())
        tmp.append(res)
    
    # Extract targets, factors, and types
    targets = [rule[0].strip() for rule in tmp]
    
    # Validate gene names
    for target in targets:
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", target):
            raise ValueError(f"Invalid gene name: {target}")
    
    factors = [rule[1].strip() for rule in tmp]
    types = [int(rule[2]) for rule in tmp]
    
    # Extract gene names from factors
    factors_tmp = [match_names(factor) for factor in factors]
    
    # Get all unique genes (order-preserving, matching R's `unique(c(...))`)
    genes = list(dict.fromkeys(targets + [gene for sublist in factors_tmp for gene in sublist]))
    
    # Initialize network structure
    fixed = {gene: -1 for gene in genes}
    interactions = {gene: [] for gene in genes}
    
    # Generate interactions
    for i, target in enumerate(targets):
        interaction = generate_fbn_interaction(factors[i], genes)
        
        if len(interaction['func']) == 1:
            fixed[target] = interaction['func'][0]
        
        interaction['type'] = types[i]
        interactions[target].append(interaction)
    
    # Handle genes with no transition functions (inputs)
    only_inputs = set(genes) - set(targets)
    if only_inputs:
        for gene in only_inputs:
            logger.warning(f'There is no transition function for gene "{gene}"! Assuming an input!')
            interactions[gene] = [{
                'input': len(interactions) + 1,
                'func': [0, 1],
                'expression': gene
            }]
    
    return {
        'interactions': interactions,
        'genes': genes,
        'fixed': fixed,
        'class': 'BooleanNetworkCollection'
    }

def generate_fbn_interaction(factor: str, genes: List[str]) -> Dict:
    """
    Generate an FBN interaction from a factor expression
    
    Args:
        factor: The factor expression string
        genes: List of all genes in the network
        
    Returns:
        A dictionary representing the interaction
    """
    # This is a placeholder - actual implementation would parse the factor expression
    # and generate the appropriate interaction structure
    input_genes = match_names(factor)  # Reuse the match_names function
    input_indices = [genes.index(gene) + 1 for gene in input_genes]  # +1 for 1-based indexing
    
    return {
        'input': input_indices,
        'expression': factor,
        'func': [0, 1],  # Default function
        'error': 0.0,
        'probability': 1.0,
        'support': 1.0,
        'timestep': 1
    }

def _as_gene_value_dict(value: Union[Dict, List], genes: List[str], default: int = -1) -> Dict[str, int]:
    """Normalize a per-gene value collection (fixed/timedecay) into a gene -> value dict."""
    if isinstance(value, dict):
        result = {gene: default for gene in genes}
        result.update(value)
        return result
    values = list(value) if value else []
    if len(values) < len(genes):
        values = [default] * len(genes)
    return dict(zip(genes, values))

def _interaction_items(interactions_for_gene: Union[Dict, List], gene_name: str) -> List[tuple]:
    """Normalize a gene's interactions (list or name->interaction dict) into (name, interaction) pairs."""
    if isinstance(interactions_for_gene, dict):
        return list(interactions_for_gene.items())
    return [(f"{gene_name}_{i+1}", interaction) for i, interaction in enumerate(interactions_for_gene)]

def convert_to_boolean_network_collection(network: Dict) -> Dict:
    """
    Convert a traditional Boolean network to fundamental Boolean network collection
    
    Args:
        network: The traditional Boolean network
        
    Returns:
        A BooleanNetworkCollection object
    """
    # if network.get('class') != 'BooleanNetwork':
    #     raise ValueError("Network must be inherited from BooleanNetwork")
    
    genes1 = network['genes']
    genes2 = network['genes']
    # Order-preserving, matching R's `unique(c(network$genes, network$genes))`;
    # this must stay identical to what's passed into merge_interaction below,
    # since the returned 'genes' list is what interaction['input'] indices
    # are decoded against.
    total_genes = list(dict.fromkeys(network['genes'] + network['genes']))
    
    # Convert interactions
    converted_interactions = {gene: convert_interaction(interaction) 
                            for gene, interaction in network['interactions'].items()}
    
    # Merge interactions
    merges = merge_interaction(converted_interactions, converted_interactions, 
                             genes1, genes2, total_genes)
    
    return {
        'interactions': merges,
        'genes': total_genes,
        'fixed': network['fixed'],
        'timedecay': {gene: 1 for gene in network['genes']},
        'class': 'BooleanNetworkCollection'
    }

def convert_interaction(interaction: Union[Dict, List]) -> List[Dict]:
    """
    Convert traditional network interactions to FBN interactions
    
    Args:
        interaction: The interaction to convert
        
    Returns:
        List of converted interactions
    """
    if isinstance(interaction, dict):
        return [interaction]
    elif isinstance(interaction, list):
        return [{
            'input': i['input'],
            'expression': i['expression'],
            'error': i.get('error', 0.0),
            'type': i.get('type', None),
            'probability': (1.0 - float(i.get('error', 0.0))),
            'support': i.get('support', None),
            'timestep': 1
        } for i in interaction]
    else:
        return []

def merge_network(network1: Dict, network2: Dict) -> Dict:
    """
    Merge two networks
    
    Args:
        network1: First network to merge
        network2: Second network to merge
        
    Returns:
        Merged FundamentalBooleanNetwork
    """
    valid_classes = {'BooleanNetworkCollection', 'FundamentalBooleanNetwork'}
    if network1.get('class') not in valid_classes:
        raise ValueError("Network1 must be inherited from FundamentalBooleanNetwork or BooleanNetworkCollection")
    if network2.get('class') not in valid_classes:
        raise ValueError("Network2 must be inherited from FundamentalBooleanNetwork or BooleanNetworkCollection")
    
    genes1 = network1['genes']
    genes2 = network2['genes']
    # Order-preserving, matching R's `unique(c(network1$genes, network2$genes))`.
    total_genes = list(dict.fromkeys(network1['genes'] + network2['genes']))
    
    # Merge interactions
    merges = merge_interaction(network1['interactions'], network2['interactions'], 
                             genes1, genes2, total_genes)
    
    # Handle fixed genes
    fixed1 = _as_gene_value_dict(network1.get('fixed', {}), genes1)
    fixed2 = _as_gene_value_dict(network2.get('fixed', {}), genes2)
    
    new_fixed = {gene: -1 for gene in total_genes}
    for gene, val in fixed1.items():
        new_fixed[gene] = val
    for gene, val in fixed2.items():
        new_fixed[gene] = val
    
    # Handle timedecay
    timedecay1 = _as_gene_value_dict(network1.get('timedecay', {}), genes1)
    timedecay2 = _as_gene_value_dict(network2.get('timedecay', {}), genes2)
    
    new_timedecay = {gene: -1 for gene in total_genes}
    for gene, val in timedecay1.items():
        new_timedecay[gene] = val
    for gene, val in timedecay2.items():
        new_timedecay[gene] = val
    
    return FundamentalBooleanNetwork({
        'interactions': merges,
        'genes': total_genes,
        'fixed': new_fixed,
        'timedecay': new_timedecay,
        'class': 'FundamentalBooleanNetwork'
    })

def merge_interaction(
    interactions1: Dict, 
    interactions2: Dict, 
    genes1: List[str], 
    genes2: List[str], 
    merged_genes: Optional[List[str]] = None
) -> Dict:
    """
    Merge network interactions
    
    Args:
        interactions1: First set of interactions
        interactions2: Second set of interactions
        genes1: Genes for first interactions
        genes2: Genes for second interactions
        merged_genes: Optional list of merged genes
        
    Returns:
        Dictionary of merged interactions
    """
    if merged_genes is None:
        merged_genes = sorted(list(set(genes1 + genes2)))
    
    res = {}
    
    # Process interactions from first group
    for name1, int_list1 in interactions1.items():
        unique_expr_1 = set()
        unique_expr_0 = set()
        a_index = 1
        i_index = 1
        
        if name1 not in res:
            res[name1] = {}
            
            for _, interaction in _interaction_items(int_list1, name1):
                input_genes1 = [genes1[i-1] for i in interaction['input']]  # Convert to 0-based
                if not input_genes1:
                    continue
                
                typ = interaction['type']
                expr = interaction['expression']
                
                # Create unique key for expression
                sorted_inputs = sorted(_utils.splitExpression(expr, 2, True))
                temp_expr = "".join(sorted_inputs)
                
                if typ == 1:
                    if temp_expr in unique_expr_1:
                        continue
                    unique_expr_1.add(temp_expr)
                else:
                    if temp_expr in unique_expr_0:
                        continue
                    unique_expr_0.add(temp_expr)
                
                if not all(g in merged_genes for g in input_genes1):
                    continue
                
                # Convert input gene names to indices in merged_genes
                new_input = [merged_genes.index(g)+1 for g in input_genes1]  # 1-based
                if not new_input:
                    continue
                
                # Create unique name for the interaction
                if typ == 1:
                    int_name = f"{name1}_{a_index}_Activator"
                    a_index += 1
                else:
                    int_name = f"{name1}_{i_index}_Inhibitor"
                    i_index += 1
                
                res[name1][int_name] = {
                    'input': new_input,
                    'expression': expr,
                    'error': interaction.get('error', 0.0),
                    'type': typ,
                    'probability': interaction.get('probability', 1.0),
                    'support': interaction.get('support', None),
                    'timestep': interaction.get('timestep', 1)
                }
    
    # Process interactions from second group that are also in first group
    for name1 in set(interactions1.keys()) & set(interactions2.keys()):
        a_index = len([k for k in res[name1].keys() if "Activator" in k]) + 1
        i_index = len([k for k in res[name1].keys() if "Inhibitor" in k]) + 1
        
        unique_expr_1 = set()
        unique_expr_0 = set()
        
        # Get existing expressions
        for int_name, interaction in res[name1].items():
            expr = interaction['expression']
            sorted_inputs = sorted(_utils.splitExpression(expr, 2, True))
            temp_expr = "".join(sorted_inputs)
            
            if interaction['type'] == 1:
                unique_expr_1.add(temp_expr)
            else:
                unique_expr_0.add(temp_expr)
        
        # Add new interactions from second group
        for _, interaction in _interaction_items(interactions2[name1], name1):
            input_genes2 = [genes2[i-1] for i in interaction['input']]  # Convert to 0-based
            if not input_genes2:
                continue
            
            typ = interaction['type']
            expr = interaction['expression']
            sorted_inputs = sorted(_utils.splitExpression(expr, 2, True))
            temp_expr = "".join(sorted_inputs)
            
            if typ == 1:
                if temp_expr in unique_expr_1:
                    continue
                unique_expr_1.add(temp_expr)
            else:
                if temp_expr in unique_expr_0:
                    continue
                unique_expr_0.add(temp_expr)
            
            if not all(g in merged_genes for g in input_genes2):
                continue
            
            new_input = [merged_genes.index(g)+1 for g in input_genes2]  # 1-based
            if not new_input:
                continue
            
            if typ == 1:
                int_name = f"{name1}_{a_index}_Activator"
                a_index += 1
            else:
                int_name = f"{name1}_{i_index}_Inhibitor"
                i_index += 1
            
            res[name1][int_name] = {
                'input': new_input,
                'expression': expr,
                'error': interaction.get('error', 0.0),
                'type': typ,
                'probability': interaction.get('probability', 1.0),
                'support': interaction.get('support', None),
                'timestep': interaction.get('timestep', 1)
            }
    
    # Process interactions from second group that aren't in first group
    names_2 = set(interactions2.keys()) - set(interactions1.keys())
    for name2 in names_2:
        if name2 not in res:
            res[name2] = {}
            
            for int_name, interaction in _interaction_items(interactions2[name2], name2):
                input_genes2 = [genes2[i-1] for i in interaction['input']]  # Convert to 0-based
                if not input_genes2:
                    continue
                
                typ = interaction['type']
                expr = interaction['expression']
                sorted_inputs = sorted(_utils.splitExpression(expr, 2, True))
                temp_expr = "".join(sorted_inputs)
                
                if not all(g in merged_genes for g in input_genes2):
                    continue
                
                new_input = [merged_genes.index(g)+1 for g in input_genes2]  # 1-based
                if not new_input:
                    continue
                
                res[name2][int_name] = {
                    'input': new_input,
                    'expression': expr,
                    'error': interaction.get('error', 0.0),
                    'type': typ,
                    'probability': interaction.get('probability', 1.0),
                    'support': interaction.get('support', None),
                    'timestep': interaction.get('timestep', 1)
                }
    
    return res

def filter_network_connections(networks: Dict) -> Dict:
    """
    Filter out non-regulating genes from the network
    
    Args:
        networks: The FBN networks to filter
        
    Returns:
        Filtered networks
    """
    genes = networks['genes']
    filtered_networks = {k: v for k, v in networks['interactions'].items() if v}
    
    regulate_genes = list(filtered_networks.keys())
    filtered_input_genes = find_all_input_genes(filtered_networks, genes)
    
    # R's `unique(c(regulategenes, filteredinputgenes))` preserves first-occurrence
    # order; a plain set() here would make gene order (and hence merge_interaction's
    # de-duplication) non-deterministic across process runs.
    mixed_genes = list(dict.fromkeys(regulate_genes + filtered_input_genes))
    
    extra_networks = {k: v for k, v in networks['interactions'].items() if k in mixed_genes}
    
    merges = merge_interaction(extra_networks, extra_networks, genes, genes, mixed_genes)
    
    # Handle fixed genes
    fixed = _as_gene_value_dict(networks.get('fixed', {}), genes)
    new_fixed = {gene: fixed.get(gene, -1) for gene in mixed_genes}
    
    # Handle timedecay
    timedecay = _as_gene_value_dict(networks.get('timedecay', {}), genes)
    new_timedecay = {gene: timedecay.get(gene, -1) for gene in mixed_genes}
    
    return FundamentalBooleanNetwork({
        'interactions': merges,
        'genes': mixed_genes,
        'fixed': new_fixed,
        'timedecay': new_timedecay,
        'class': 'FundamentalBooleanNetwork'
    })

def find_all_input_genes(network_interactions: Dict, genes: List[str]) -> List[str]:
    """
    Find all input genes from network interactions
    
    Args:
        network_interactions: The network interactions
        genes: List of all genes
        
    Returns:
        List of input genes
    """
    input_indices = set()
    for interactions in network_interactions.values():
        # interactions may be a plain list (freshly mined) or a dict keyed by
        # interaction name (already merged/filtered) - support both shapes.
        items = interactions.values() if isinstance(interactions, dict) else interactions
        for interaction in items:
            input_indices.update(interaction['input'])
    
    return [genes[i-1] for i in input_indices]  # Convert to 0-based

def find_all_target_genes(network_interactions: Dict) -> List[str]:
    """
    Find all target genes from network interactions
    
    Args:
        network_interactions: The network interactions
        
    Returns:
        List of target genes
    """
    return [gene for gene, interactions in network_interactions.items() if interactions]

def filter_network_connections_by_genes(
    networks: Dict,
    genelist: List[str],
    exclusive: bool = True,
    expand: bool = False,
) -> Dict:
    """
    Filter a network down to the genes related to (or excluding) a gene list.

    Args:
        networks: The Fundamental Boolean Network
        genelist: The genes to filter by
        exclusive: If True, keep everything except genelist's own regulations;
            if False, keep only genelist's own regulations
        expand: If True, also pull in the regulators of the kept genes

    Returns:
        A filtered FundamentalBooleanNetwork
    """
    if not genelist:
        raise ValueError("The genelist is empty")

    genes = networks['genes']
    if exclusive:
        filtered_networks = {k: v for k, v in networks['interactions'].items() if k not in genelist}
    else:
        filtered_networks = {k: v for k, v in networks['interactions'].items() if k in genelist}

    regulate_genes = list(filtered_networks.keys())
    filtered_input_genes = find_all_input_genes(filtered_networks, genes)

    if not expand:
        extra_networks = filtered_networks
        mixed_genes = filtered_input_genes
    else:
        mixed_genes = list(set(regulate_genes + filtered_input_genes))
        extra_networks = {k: v for k, v in networks['interactions'].items() if k in mixed_genes}
        mixed_genes = find_all_input_genes(extra_networks, genes)

    merges = merge_interaction(extra_networks, extra_networks, genes, genes, mixed_genes)

    fixed = _as_gene_value_dict(networks.get('fixed', {}), genes)
    new_fixed = {gene: fixed.get(gene, -1) for gene in mixed_genes}

    timedecay = _as_gene_value_dict(networks.get('timedecay', {}), genes)
    new_timedecay = {gene: timedecay.get(gene, -1) for gene in mixed_genes}

    res = {
        'interactions': merges,
        'genes': mixed_genes,
        'fixed': new_fixed,
        'timedecay': new_timedecay,
        'class': 'FundamentalBooleanNetwork',
    }
    return filter_network_connections(res)

def filter_network_connections_by_input_genes(networks: Dict, genelist: List[str]) -> Dict:
    """
    Filter a network down to only the given genes that act as inputs/regulators.

    Args:
        networks: The Fundamental Boolean Network
        genelist: Candidate input genes to keep

    Returns:
        A filtered FundamentalBooleanNetwork
    """
    if not genelist:
        raise ValueError("The genelist is empty")

    genes = networks['genes']
    filtered_networks = networks['interactions']

    filtered_input_genes = find_all_input_genes(filtered_networks, genes)
    mixed_genes = [g for g in genelist if g in filtered_input_genes]
    if not mixed_genes:
        raise ValueError("No input genes found")

    merges = merge_interaction(filtered_networks, filtered_networks, genes, genes, mixed_genes)

    fixed = _as_gene_value_dict(networks.get('fixed', {}), genes)
    new_fixed = {gene: fixed.get(gene, -1) for gene in mixed_genes}

    timedecay = _as_gene_value_dict(networks.get('timedecay', {}), genes)
    new_timedecay = {gene: timedecay.get(gene, -1) for gene in mixed_genes}

    res = {
        'interactions': merges,
        'genes': mixed_genes,
        'fixed': new_fixed,
        'timedecay': new_timedecay,
        'class': 'FundamentalBooleanNetwork',
    }
    return filter_network_connections(res)

def find_all_backward_related_genes(
    networks: Dict,
    target_gene: str,
    regulation_type: Optional[int] = None,
    target_type: Optional[int] = None,
    max_deep: int = 1,
    next_level_mix_type: bool = False,
) -> Dict:
    """
    Find the (possibly transitive) upstream regulators of target_gene.

    Args:
        networks: The Fundamental Boolean Network
        target_gene: The target gene
        regulation_type: 1 (activation) or 0 (inhibition) to filter by, or None for both
        target_type: Unused (kept for parity with the R signature)
        max_deep: How many layers of indirection to drill down
        next_level_mix_type: If True, expansion at deeper levels includes all regulation types

    Returns:
        A filtered FundamentalBooleanNetwork of upstream regulators
    """
    prepare_network = filter_network_connections_by_genes(networks, [target_gene], exclusive=False, expand=False)

    for gene_name, interactions in list(prepare_network['interactions'].items()):
        kept = {
            name: interaction
            for name, interaction in _interaction_items(interactions, gene_name)
            if regulation_type is None or interaction['type'] == regulation_type
        }
        prepare_network['interactions'][gene_name] = kept

    prepare_network = filter_network_connections(prepare_network)

    if max_deep > 1 and prepare_network['interactions']:
        expand_genes = [g for g in prepare_network['genes'] if g != target_gene]
        for this_target_gene in expand_genes:
            new_network = find_all_backward_related_genes(
                networks, this_target_gene,
                regulation_type=regulation_type,
                max_deep=max_deep - 1,
                next_level_mix_type=next_level_mix_type,
            )
            prepare_network = filter_network_connections(merge_network(prepare_network, new_network))

    return prepare_network

def find_backward_related_network_by_genes(
    networks: Dict,
    target_gene_list: List[str],
    regulation_type: Optional[int] = None,
    max_deep: int = 1,
    next_level_mix_type: bool = False,
) -> Dict:
    """
    Find the upstream regulators for a list of target genes.

    Args:
        networks: The Fundamental Boolean Network
        target_gene_list: The target genes
        regulation_type: 1 (activation) or 0 (inhibition) to filter by, or None for both
        max_deep: How many layers of indirection to drill down
        next_level_mix_type: If True, expansion at deeper levels includes all regulation types

    Returns:
        A filtered FundamentalBooleanNetwork of upstream regulators
    """
    if not target_gene_list:
        raise ValueError("The genelist is empty")

    prepare_network = None
    for target_gene in target_gene_list:
        new_network = find_all_backward_related_genes(
            networks, target_gene,
            regulation_type=regulation_type,
            max_deep=max_deep,
            next_level_mix_type=next_level_mix_type,
        )
        prepare_network = new_network if prepare_network is None else merge_network(prepare_network, new_network)
        prepare_network = filter_network_connections(prepare_network)

    return prepare_network

def find_all_forward_related_genes(
    networks: Dict,
    target_gene: str,
    regulation_type: Optional[int] = None,
    target_type: Optional[int] = None,
    main_target_gene: Optional[str] = None,
    main_target_type: Optional[int] = None,
    max_deep: int = 1,
    next_level_mix_type: bool = False,
) -> Dict:
    """
    Find the (possibly transitive) downstream targets regulated by target_gene.

    Args:
        networks: The Fundamental Boolean Network
        target_gene: The gene whose downstream targets are searched for
        regulation_type: 1 (activation) or 0 (inhibition) to filter by, or None for both
        target_type: 1 if target_gene must activate (no "!"), 0 if it must inhibit, or None for both
        main_target_gene: The original gene the search started from (defaults to target_gene)
        main_target_type: The connection type expected for main_target_gene (defaults to target_type)
        max_deep: How many layers of indirection to drill down
        next_level_mix_type: If True, expansion at deeper levels includes all regulation types

    Returns:
        A filtered FundamentalBooleanNetwork of downstream targets
    """
    genes = networks['genes']
    if main_target_gene is None:
        main_target_gene = target_gene
    if main_target_type is None:
        main_target_type = target_type

    prepare_network = {
        'interactions': dict(networks['interactions']),
        'genes': genes,
        'fixed': networks.get('fixed', {}),
        'timedecay': networks.get('timedecay', {}),
        'class': networks.get('class', 'FundamentalBooleanNetwork'),
    }

    for gene_name, interactions in networks['interactions'].items():
        kept = {}
        for name, interaction in _interaction_items(interactions, gene_name):
            input_genes = [genes[i - 1] for i in interaction['input']]
            typ = interaction['type']
            expression = interaction['expression']

            if target_gene not in input_genes:
                continue
            if regulation_type is not None and typ != regulation_type:
                continue

            this_target_type = 0 if f"!{target_gene}" in expression else 1
            this_main_target_type = 0 if f"!{main_target_gene}" in expression else 1

            if (main_target_gene != target_gene and main_target_gene in input_genes
                    and this_main_target_type != main_target_type):
                continue
            if target_type is not None and this_target_type != target_type:
                continue

            kept[name] = interaction

        prepare_network['interactions'][gene_name] = kept

    prepare_network = filter_network_connections(prepare_network)

    if max_deep > 1 and prepare_network['interactions']:
        filtered = {k: v for k, v in prepare_network['interactions'].items() if v}
        expand_genes = [g for g in find_all_target_genes(filtered) if g != target_gene]
        next_max_deep = max_deep - 1
        next_target_type = 1 if regulation_type == 1 else 0

        for this_target_gene in expand_genes:
            regulation_types_to_expand = [1, 0] if next_level_mix_type else [regulation_type]
            for next_regulation_type in regulation_types_to_expand:
                new_network = find_all_forward_related_genes(
                    networks, this_target_gene,
                    regulation_type=next_regulation_type, target_type=next_target_type,
                    main_target_gene=main_target_gene, main_target_type=main_target_type,
                    max_deep=next_max_deep, next_level_mix_type=next_level_mix_type,
                )
                prepare_network = filter_network_connections(merge_network(prepare_network, new_network))

    return prepare_network

def find_forward_related_network_by_genes(
    networks: Dict,
    target_gene_list: List[str],
    regulation_type: Optional[int] = None,
    target_type: Optional[int] = None,
    max_deep: int = 1,
    next_level_mix_type: bool = False,
) -> Dict:
    """
    Find the downstream targets regulated by a list of genes.

    Args:
        networks: The Fundamental Boolean Network
        target_gene_list: The target genes
        regulation_type: 1 (activation) or 0 (inhibition) to filter by, or None for both
        target_type: 1 if target genes must activate, 0 if they must inhibit, or None for both
        max_deep: How many layers of indirection to drill down
        next_level_mix_type: If True, expansion at deeper levels includes all regulation types

    Returns:
        A filtered FundamentalBooleanNetwork of downstream targets
    """
    if not target_gene_list:
        raise ValueError("The genelist is empty")

    prepare_network = None
    for target_gene in target_gene_list:
        new_network = find_all_forward_related_genes(
            networks, target_gene,
            regulation_type=regulation_type, target_type=target_type,
            main_target_gene=target_gene, main_target_type=target_type,
            max_deep=max_deep, next_level_mix_type=next_level_mix_type,
        )
        prepare_network = new_network if prepare_network is None else merge_network(prepare_network, new_network)
        prepare_network = filter_network_connections(prepare_network)

    return prepare_network
