import re
from typing import List, Dict, Any, Optional, Union, Set
import numpy as np
import fbnnet_utils
# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def match_names(rule: str) -> List[str]:
        """Extract gene names from a rule expression"""
        regex = r"([_a-zA-Z][_a-zA-Z0-9]*)[,| |\)|\||\&|\[|\]]"
        rule = re.sub(r"\s+", "", rule) + " "
        matches = re.findall(regex, rule)
        genes = []
        
        for m in matches:
            # The regex pattern has a group, so we get the first group
            gene = re.match(regex, m)
            if gene:
                genes.append(gene.group(1))
        
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
    
    # Get all unique genes
    genes = list(set(targets + [gene for sublist in factors_tmp for gene in sublist]))
    
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
    total_genes = network['genes']
    
    # Convert interactions
    converted_interactions = {gene: convert_interaction(interaction) 
                            for gene, interaction in network['interactions'].items()}
    
    # Merge interactions
    merges = merge_interaction(converted_interactions, converted_interactions, 
                             genes1, genes2, total_genes)
    
    return {
        'interactions': merges,
        'genes': list(set(network['genes'] + network['genes'])),
        'fixed': network['fixed'],
        'timedecay': [1] * len(network['genes']),
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
    total_genes = list(set(network1['genes'] + network2['genes']))
    
    # Merge interactions
    merges = merge_interaction(network1['interactions'], network2['interactions'], 
                             genes1, genes2, total_genes)
    
    # Handle fixed genes
    fixed1 = network1.get('fixed', [-1] * len(genes1))
    if len(fixed1) < len(genes1):
        fixed1 = [-1] * len(genes1)
    fixed1 = dict(zip(genes1, fixed1))
    
    fixed2 = network2.get('fixed', [-1] * len(genes2))
    if len(fixed2) < len(genes2):
        fixed2 = [-1] * len(genes2)
    fixed2 = dict(zip(genes2, fixed2))
    
    new_fixed = [-1] * len(total_genes)
    new_fixed = dict(zip(total_genes, new_fixed))
    for gene, val in fixed1.items():
        new_fixed[gene] = val
    for gene, val in fixed2.items():
        new_fixed[gene] = val
    
    # Handle timedecay
    timedecay1 = network1.get('timedecay', [-1] * len(genes1))
    if len(timedecay1) < len(genes1):
        timedecay1 = [-1] * len(genes1)
    timedecay1 = dict(zip(genes1, timedecay1))
    
    timedecay2 = network2.get('timedecay', [-1] * len(genes2))
    if len(timedecay2) < len(genes2):
        timedecay2 = [-1] * len(genes2)
    timedecay2 = dict(zip(genes2, timedecay2))
    
    new_timedecay = [-1] * len(total_genes)
    new_timedecay = dict(zip(total_genes, new_timedecay))
    for gene, val in timedecay1.items():
        new_timedecay[gene] = val
    for gene, val in timedecay2.items():
        new_timedecay[gene] = val
    
    return {
        'interactions': merges,
        'genes': total_genes,
        'fixed': new_fixed,
        'timedecay': new_timedecay,
        'class': 'FundamentalBooleanNetwork'
    }

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
            
            for j, interaction in enumerate(int_list1):
                input_genes1 = [genes1[i-1] for i in interaction['input']]  # Convert to 0-based
                if not input_genes1:
                    continue
                
                typ = interaction['type']
                expr = interaction['expression']
                
                # Create unique key for expression
                sorted_inputs = sorted(fbnnet_utils.splitExpression(expr, 2, True))
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
            sorted_inputs = sorted(fbnnet_utils.splitExpression(expr, 2, True))
            temp_expr = "".join(sorted_inputs)
            
            if interaction['type'] == 1:
                unique_expr_1.add(temp_expr)
            else:
                unique_expr_0.add(temp_expr)
        
        # Add new interactions from second group
        for j, interaction in enumerate(interactions2[name1]):
            input_genes2 = [genes2[i-1] for i in interaction['input']]  # Convert to 0-based
            if not input_genes2:
                continue
            
            typ = interaction['type']
            expr = interaction['expression']
            sorted_inputs = sorted(fbnnet_utils.splitExpression(expr, 2, True))
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
            
            for j, interaction in enumerate(interactions2[name2]):
                input_genes2 = [genes2[i-1] for i in interaction['input']]  # Convert to 0-based
                if not input_genes2:
                    continue
                
                typ = interaction['type']
                expr = interaction['expression']
                sorted_inputs = sorted(fbnnet_utils.splitExpression(expr, 2, True))
                temp_expr = "".join(sorted_inputs)
                
                if not all(g in merged_genes for g in input_genes2):
                    continue
                
                new_input = [merged_genes.index(g)+1 for g in input_genes2]  # 1-based
                if not new_input:
                    continue
                
                int_name = list(interactions2[name2].keys())[j]
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

# def split_expression(expr: str, split_by: str = ",", strip: bool = True) -> List[str]:
#     """Split an expression string into parts"""
#     if not expr:
#         return []
#     parts = expr.split(split_by)
#     if strip:
#         parts = [p.strip() for p in parts]
#     return parts

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
    
    mixed_genes = list(set(regulate_genes + filtered_input_genes))
    
    extra_networks = {k: v for k, v in networks['interactions'].items() if k in mixed_genes}
    
    merges = merge_interaction(extra_networks, extra_networks, genes, genes, mixed_genes)
    
    # Handle fixed genes
    fixed = networks.get('fixed', [-1] * len(genes))
    if len(fixed) < len(genes):
        fixed = [-1] * len(genes)
    fixed = dict(zip(genes, fixed))
    
    new_fixed = [-1] * len(mixed_genes)
    new_fixed = dict(zip(mixed_genes, new_fixed))
    for gene in mixed_genes:
        if gene in fixed:
            new_fixed[gene] = fixed[gene]
    
    # Handle timedecay
    timedecay = networks.get('timedecay', [-1] * len(genes))
    if len(timedecay) < len(genes):
        timedecay = [-1] * len(genes)
    timedecay = dict(zip(genes, timedecay))
    
    new_timedecay = [-1] * len(mixed_genes)
    new_timedecay = dict(zip(mixed_genes, new_timedecay))
    for gene in mixed_genes:
        if gene in timedecay:
            new_timedecay[gene] = timedecay[gene]
    
    return {
        'interactions': merges,
        'genes': mixed_genes,
        'fixed': new_fixed,
        'timedecay': new_timedecay,
        'class': 'FundamentalBooleanNetwork'
    }

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

# Example usage
# if __name__ == "__main__":
#     # Example FBN network file (comma-separated)
#     example_file = """targets,factors,type
# Gene1,Gene2 & Gene3,1
# Gene2,!Gene1 | Gene3,0
# Gene3,Gene1,1"""
    
#     # Write to a temporary file
#     with open("example_fbn.csv", "w") as f:
#         f.write(example_file)
    
#     # Load the network
#     network = load_network("example_fbn.csv")
#     print("Loaded network:")
#     print(network)
    
#     # Convert to BooleanNetworkCollection
#     bn_collection = convert_to_boolean_network_collection(network)
#     print("\nConverted to BooleanNetworkCollection:")
#     print(bn_collection)
    
#     # Merge with itself
#     merged = merge_network(network, network)
#     print("\nMerged network:")
#     print(merged)
    
#     # Filter network connections
#     filtered = filter_network_connections(merged)
#     print("\nFiltered network:")
#     print(filtered)