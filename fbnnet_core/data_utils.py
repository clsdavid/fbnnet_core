import numpy as np
import pandas as pd
from itertools import product

def generateBoolNetTimeseries(network, initialStates, numMeasurements, 
                            transition_type="synchronous", geneProbabilities=None):
    """
    Python equivalent of R's genereateBoolNetTimeseries function
    
    Args:
        network: Dictionary representing the Boolean network (from load_network)
        initialStates: List of initial state dictionaries
        numMeasurements: Number of time steps to simulate
        transition_type: "synchronous", "asynchronous", or "probabilistic"
        geneProbabilities: Optional probabilities for probabilistic transitions
    
    Returns:
        List of 2D numpy arrays with gene rows and timepoint columns
    """
    results = []
    
    for state in initialStates:
        # Convert initial state to array format
        genes = network['genes']
        time_series = np.zeros((len(genes), numMeasurements), dtype=int)
        
        # Set initial state
        current_state = np.array([state[gene] for gene in genes])
        time_series[:, 0] = current_state
        
        # Simulate time series
        for t in range(1, numMeasurements):
            current_state = stateTransition(network, current_state, 
                                         transition_type, geneProbabilities)
            time_series[:, t] = current_state
        
        # Add row and column names to match R output
        time_series = np.array(time_series)
        time_series = np.reshape(time_series, (len(genes), numMeasurements))
        
        results.append(time_series)
    
    return results

def stateTransition(network, state, transition_type="synchronous", geneProbabilities=None):
    """
    Python equivalent of BoolNet::stateTransition()
    """
    new_state = np.zeros_like(state)
    genes = network['genes']
    
    if transition_type == "synchronous":
        # Synchronous update all genes at once
        for i, gene in enumerate(genes):
            expr = network['expressions'][gene]
            new_state[i] = evaluateBooleanExpression(expr, state, genes)
    
    elif transition_type == "asynchronous":
        # Asynchronous update - randomly select one gene to update
        new_state = state.copy()
        idx = np.random.randint(len(genes))
        gene = genes[idx]
        expr = network['expressions'][gene]
        new_state[idx] = evaluateBooleanExpression(expr, state, genes)
    
    elif transition_type == "probabilistic":
        # Probabilistic update
        new_state = state.copy()
        for i, gene in enumerate(genes):
            if geneProbabilities and gene in geneProbabilities:
                if np.random.random() < geneProbabilities[gene]:
                    expr = network['expressions'][gene]
                    new_state[i] = evaluateBooleanExpression(expr, state, genes)
    
    return new_state

def evaluateBooleanExpression(expr, state, genes):
    """
    Evaluates a Boolean expression given current state
    """
    # Create a namespace with gene values
    namespace = {gene: bool(val) for gene, val in zip(genes, state)}
    namespace['__builtins__'] = None  # For security
    
    # Convert Boolean operators to Python syntax
    expr = expr.replace('&', ' and ').replace('|', ' or ').replace('!', ' not ')
    
    try:
        result = eval(expr, namespace)
        return 1 if result else 0
    except:
        return 0



def generateAllCombinationBinary(genelist=None, begin=1, last=0):
    """
    Python equivalent of R's generateAllCombinationBinary function
    Generates all binary combinations of genes in the genelist
    
    Args:
        genelist (list): List of gene names
        begin (int): Starting index (1-based)
        last (int): Ending index (0 means generate all)
    
    Returns:
        list: List of dictionaries representing each binary combination
    """
    if not genelist:
        raise ValueError("The genelist is empty")
    
    n = len(genelist)
    total = 2 ** n
    
    if last == 0:
        last = total
    
    # Adjust for Python's 0-based indexing vs R's 1-based
    begin = begin - 1
    last = last
    
    # Generate all binary combinations
    binary_combinations = list(product([0, 1], repeat=n))
    
    # Slice the requested range
    combinations = binary_combinations[begin:last]
    
    # Convert to list of dictionaries with gene names as keys
    result = []
    for combo in combinations:
        result.append(dict(zip(genelist, combo)))
    
    return result


def dividedVectorIntoSmallgroups(vector, maxElements=20):
    """
    Python equivalent of R's dividedVectorIntoSmallgroups function.
    Splits a vector into consecutive chunks of at most maxElements items.

    Args:
        vector (list): The vector to split
        maxElements (int): The max number of elements per group

    Returns:
        dict: {"clusters": [list, ...], "original": vector}
    """
    if not isinstance(vector, (list, tuple, np.ndarray)):
        raise ValueError("The parameter 'vector' must be a vector")

    vector = list(vector)
    clusters = [vector[i:i + maxElements] for i in range(0, len(vector), maxElements)]

    return {"clusters": clusters, "original": vector}


def getRelatedGeneTimeseries(timeseries, genelist=None):
    """
    Python equivalent of R's getRelatedGeneTimeseries function.
    Reduces each sample in a timeseries cube down to the rows (genes) in genelist.

    Args:
        timeseries (list): A list of samples (pandas DataFrame or 2D numpy array)
            with genes on rows and time steps on columns
        genelist (list): The genes to keep

    Returns:
        list: The same samples, each filtered down to genelist's rows
    """
    genelist = genelist or []
    result = []
    for sheet in timeseries:
        if isinstance(sheet, pd.DataFrame):
            result.append(sheet.loc[sheet.index.isin(genelist)])
        else:
            raise ValueError("Each sample must be a pandas DataFrame with gene names as its index")
    return result


def randomGenerateBinary(genelist=None, maxState=0):
    """
    Python equivalent of R's randomGenerateBinary function.
    Randomly generates binary (0/1) states for the genes in genelist.

    Args:
        genelist (list): List of gene names
        maxState (int): The number of random states to generate (0 means 2^len(genelist))

    Returns:
        list: A list of dictionaries, each mapping gene name -> 0/1
    """
    if not genelist:
        raise ValueError("The genelist is empty")

    if maxState == 0:
        maxState = 2 ** len(genelist)

    result = []
    for _ in range(maxState):
        result.append({gene: int(np.random.random() < 0.5) for gene in genelist})

    return result
