import re
from typing import List, Dict, Union, Any
from .general_utils import dissolve
from .fbn_types import FundamentalBooleanNetwork
import fbnnet_utils

def remove_duplicates(factors: Union[List[Dict], str]) -> List[Dict]:
    """Remove duplicate factors based on identity.
    
    Args:
        factors: Either a list of dictionaries (each with an "identity" key) or a string
        
    Returns:
        List of unique dictionaries based on their identity
    """
    # Handle case where input is a string
    if isinstance(factors, str):
        return []
    
    seen = set()
    unique_factors = []
    for factor in factors:
        # Ensure factor is a dictionary
        if not isinstance(factor, dict):
            continue
            
        identity = factor.get("identity", None)
        if identity not in seen:
            seen.add(identity)
            unique_factors.append(factor)
    return unique_factors

def is_atom_node(sub_expression: List[str]) -> bool:
    """
    Check if an expression is an atomic node
    
    Args:
        sub_expression: A sub Boolean expression
        
    Returns:
        bool: True if atomic node, False otherwise
    """
    unwant_list = {"|", "&", "(", "[", "{", ")", "]", "}"}
    for char in sub_expression:
        if char in unwant_list:
            return False
    return True

def is_sub_expression(sub_expression: List[str]) -> bool:
    """
    Check if an expression is a sub-expression
    
    Args:
        sub_expression: A sub Boolean expression
        
    Returns:
        bool: True if valid sub-expression, False otherwise
    """
    group_a = {"(", "[", "{"}
    group_b = {")", "]", "}"}
    start_bracket = 0
    end_bracket = 0
    num_of_bracket = 0
    
    for i, char in enumerate(sub_expression, 1):
        if char in group_a:
            if num_of_bracket == 0:
                start_bracket = i
            num_of_bracket += 1
            continue
        
        if char in group_b:
            if num_of_bracket == 1:
                end_bracket = i
            num_of_bracket -= 1
    
    return start_bracket == 1 and end_bracket == len(sub_expression)

def remove_first_level_bracket(splitted_expression: List[str]) -> List[str]:
    """
    Remove the first level brackets from an expression
    
    Args:
        splitted_expression: A splitted Boolean expression
        
    Returns:
        List[str]: Expression with outer brackets removed
    """
    if is_sub_expression(splitted_expression):
        return splitted_expression[1:-1]
    return splitted_expression

def is_applied_de_morgan_law(splitted_expression: List[str]) -> bool:
    """
    Check if De Morgan's law is applied to an expression
    
    Args:
        splitted_expression: A splitted Boolean expression
        
    Returns:
        bool: True if De Morgan's law is applied
    """
    group_a = {"(", "[", "{"}
    group_b = {")", "]", "}"}
    
    if not splitted_expression or splitted_expression[0] != "!":
        return False
    
    sub_expression = splitted_expression[1:]
    start_bracket = 0
    end_bracket = 0
    num_of_bracket = 0
    
    for i, char in enumerate(sub_expression, 1):
        if char in group_a:
            if num_of_bracket == 0:
                start_bracket = i
            num_of_bracket += 1
            continue
        
        if char in group_b:
            if num_of_bracket == 1:
                end_bracket = i
            num_of_bracket -= 1
    
    return start_bracket == 1 and end_bracket == len(sub_expression)

def flat_de_morgan_law(splitted_expression: List[str]) -> List[Any]:
    """
    Flatten an expression with De Morgan's law applied
    
    Args:
        splitted_expression: A splitted Boolean expression
        
    Returns:
        List[Any]: Flattened expression
    """
    if not is_applied_de_morgan_law(splitted_expression):
        return splitted_expression
    
    negation = splitted_expression[0]
    part_of_expression = splitted_expression[1:]
    split_exp = remove_first_level_bracket(part_of_expression)
    res = convert_into_expression_tree(split_exp)
    
    if "&" in split_exp and "|" not in split_exp:
        for i in range(len(res)):
            if res[i] == "&":
                res[i] = ["|"]
            else:
                res[i] = [negation] + res[i]
    elif "|" in split_exp and "&" not in split_exp:
        for i in range(len(res)):
            if res[i] == "|":
                res[i] = ["&"]
            else:
                res[i] = [negation] + res[i]
    
    return res

def convert_into_expression_tree(splitted_expression: List[str]) -> List[Any]:
    """
    Convert a splitted Boolean expression into an expression tree
    
    Args:
        splitted_expression: A splitted Boolean expression
        
    Returns:
        List[Any]: Expression tree structure
    """
    split_exp = remove_first_level_bracket(splitted_expression)
    operators = {"|", "&"}
    group_a = {"(", "[", "{"}
    group_b = {")", "]", "}"}
    
    if len(split_exp) == 1:
        return [split_exp[0]]
    
    if not ("&" in split_exp or "|" in split_exp):
        return split_exp
    
    contained_bracket = 0
    res = []
    left_cut = 0
    
    for i, char in enumerate(split_exp):
        if char in group_a:
            contained_bracket += 1
            continue
        
        if char in group_b:
            contained_bracket -= 1
        
        if contained_bracket > 0:
            continue
        
        if char in operators:
            expression_left = split_exp[left_cut:i]
            expression_right = split_exp[i+1:]
            
            if is_sub_expression(expression_left):
                res.append(convert_into_expression_tree(expression_left))
            else:
                res.append(expression_left)
            
            res.append(char)
            
            if is_sub_expression(expression_right):
                res.append(convert_into_expression_tree(expression_right))
            else:
                if is_atom_node(expression_right):
                    res.append(expression_right)
                    break
                
                if is_applied_de_morgan_law(expression_right):
                    res.append(flat_de_morgan_law(expression_right))
                    break
                
                left_cut = i + 1
    
    return res

def construct_fbn_functions(expression_tree: List[Any]) -> List[List[Any]]:
    """
    Construct FBN functions from an expression tree
    
    Args:
        expression_tree: An expression tree
        
    Returns:
        List[List[Any]]: Constructed FBN functions
    """
    stem = expression_tree
    operators = {"|", "&"}
    res = [[]]
    index = 0
    new_line = False
    
    for item in stem:
        if not (isinstance(item, str) and len(item) == 1 and item[0] in operators):
            if isinstance(item, list):
                sub_res = construct_fbn_functions(item)
                
                if len(res) > index:
                    pre = res[index]
                else:
                    pre = []
                
                for j, sub_item in enumerate(sub_res):
                    if new_line:
                        if index < len(res):
                            res[index] = sub_item
                        else:
                            res.append(sub_item)
                    else:
                        if pre:
                            res[index] = pre + sub_item
                        else:
                            res[index] = sub_item
                    
                    if j < len(sub_res) - 1:
                        index += 1
                        res.append([])
            else:
                if index == len(res):
                    res.append([item])
                else:
                    if len(res) > 1:
                        for r in res:
                            r.append(item)
                    else:
                        res[index].append(item)
        else:
            if isinstance(item, str) and item == "|":
                index += 1
                res.append([])
                new_line = True
            elif isinstance(item, str) and item == "&":
                if len(res) > 1:
                    for r in res:
                        r.append("&")
                else:
                    res[index].append("&")
                new_line = False
    
    return res

def generate_fbn_interaction(expression_string: str, genes: List[str]) -> Dict[str, Any]:
    """
    Generate FBN interaction
    
    Args:
        expression_string: An expression string
        genes: The involved genes
        
    Returns:
        Dict[str, Any]: FBN interaction dictionary
    """
    res = {}
    splitted_expression = fbnnet_utils.splitExpression(expression_string, 1, False)
    gene_inputs = [i+1 for i, gene in enumerate(genes) if gene in splitted_expression]
    res["input"] = gene_inputs
    res["expression"] = expression_string
    return res

def regenerate_interactions(name: str, expression_string: str, genes: List[str], 
                          error: float, interaction_type: int, 
                          probability: float = None, support: float = None, 
                          timestep: int = 1) -> List[Dict[str, Any]]:
    """
    Reconstruct FBN interactions
    
    Args:
        name: The name of the interaction
        expression_string: The expression string
        genes: The involved genes
        error: The error value
        interaction_type: The type of interaction
        probability: The probability
        support: The support threshold
        timestep: The time step
        
    Returns:
        List[Dict[str, Any]]: List of reconstructed interactions
    """
    if isinstance(genes, list) and all(isinstance(g, list) for g in genes):
        gene_list = [g for sublist in genes for g in sublist]
    else:
        gene_list = genes
    
    split = fbnnet_utils.splitExpression(expression_string, 1, False)
    tree = convert_into_expression_tree(split)
    functions = construct_fbn_functions(tree)
    res = []
    
    for i, func in enumerate(functions):
        func_str = "".join(str(x) for x in func)
        fbn_interaction = generate_fbn_interaction(func_str, gene_list)
        interaction = {
            "input": fbn_interaction["input"],
            "expression": fbn_interaction["expression"],
            "error": error if "error" not in fbn_interaction else fbn_interaction["error"],
            "type": interaction_type if interaction_type in {0, 1} else 1,
            "probability": probability if probability is not None else "NA",
            "support": support if support is not None else "NA",
            "timestep": timestep if timestep is not None else "NA"
        }
        
        typename = "Activator" if interaction["type"] == 1 else "Inhibitor"
        if len(functions) == 1:
            interaction_name = f"{name}_{typename}"
        else:
            interaction_name = f"{name}_{i}_{typename}"
        
        res.append({interaction_name: interaction})
    
    return res

def convert_mined_result_to_fbn_network(miner_result: Dict[str, Any], genes: List[str]) -> Dict[str, Any]:
    """
    Convert mined result to FBN network
    
    Args:
        miner_result: The result of mining
        genes: The genes involved
        
    Returns:
        Dict[str, Any]: FBN network structure
    """
    try:
        res = {
            "genes": genes,
            "interactions": {},
            "fixed": {gene: -1 for gene in genes},
            "timedecay": {gene: 1 for gene in genes}
        }
        
        entry = {}
        
        for name in genes:
            entry[name] = {}
            verify_items = miner_result.get(name, [])
            
            if verify_items:
                interaction_items = dissolve(miner_result[name])
                
                for j, item in enumerate(interaction_items, 1):
                    expression = item['factor']
                    error = item['error']
                    probability = item['P']
                    support = item['support']
                    item_type = int(item['type'])
                    timestep = item['timestep']
                    
                    interactions = regenerate_interactions(
                        f"{name}_{j}", expression, genes, error, item_type,
                        probability, support, timestep
                    )
                    
                    for interaction in interactions:
                        for int_name, int_data in interaction.items():
                            entry[name][int_name] = int_data
        
        res["interactions"] = entry
        return res
    
    except Exception as e:
        raise ValueError(f"Error converting to FBN Network: {str(e)}")

def convert_to_fbn_network(network: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a boolean network to FBN network
    
    Args:
        network: A boolean network object
        
    Returns:
        Dict[str, Any]: FBN network structure
    """
    if network.get("class") != "BooleanNetworkCollection":
        raise ValueError("Network must be inherited from BooleanNetwork")

    try:
        res = {
            "genes": network["genes"],
            "interactions": {},
            "fixed": network.get("fixed", {gene: -1 for gene in network["genes"]}),
            "timedecay": {gene: 1 for gene in network["genes"]}
        }
        
        entry = {}
        
        for name, interaction_items in network["interactions"].items():
            entry[name] = {}
            
            for j, item in enumerate(interaction_items, 1):
                expression = item["expression"]
                error = item.get("error", 0)
                probability = 1 - float(error) if error else 1
                timestep = 1
                support = 1
                item_type = item.get("type", None)
                
                interactions = regenerate_interactions(
                    f"{name}_{j}", expression, network["genes"], error, item_type,
                    probability, support, timestep
                )
                
                for interaction in interactions:
                    for int_name, int_data in interaction.items():
                        entry[name][int_name] = int_data
        
        res["interactions"] = entry
        res["class"] = "FundamentalBooleanNetwork"
        return FundamentalBooleanNetwork(res)
    
    except Exception as e:
        raise ValueError(f"Error converting to FBN Network: {str(e)}")

if __name__ == "__main__":
    # Test atomic nodes
    print("\n1. Testing atomic nodes:")
    print(is_atom_node(['A', 'B']))       # True
    print(is_atom_node(['A', '&']))       # False
    
    # Test De Morgan's Law
    print("\n2. Testing De Morgan's Law:")
    dm_expr = ['!', '(', 'A', '&', 'B', ')']
    print(is_applied_de_morgan_law(dm_expr))  # True
    print(flat_de_morgan_law(dm_expr))       # Flattened version
    
    # Test full pipeline
    print("\n3. Testing full pipeline:")
    expr = "!(A&B)"
    genes = ["A", "B", "C"]
    split_expr = fbnnet_utils.splitExpression(expr, 1, False)
    tree = convert_into_expression_tree(split_expr)
    fbn_funcs = construct_fbn_functions(tree)
    interactions = regenerate_interactions("Gene1", expr, genes, 0.1, 1)
    
    print("Original:", expr)
    print("Split:", split_expr)
    print("Tree:", tree)
    print("Functions:", fbn_funcs)
    print("Interactions:", interactions)