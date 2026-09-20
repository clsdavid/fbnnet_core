import re
from collections import defaultdict

def load_network(file_path):
    """
    Python equivalent of R's BoolNet::loadNetwork()
    Reads a .bn file and returns a dictionary representing the Boolean network
    
    Args:
        file_path (str): Path to the .bn file
    
    Returns:
        dict: A dictionary representing the Boolean network with:
              - 'genes': List of gene names
              - 'interactions': Dictionary of gene interactions
              - 'expressions': Dictionary of Boolean expressions
    """
    network = {
        'genes': [],
        'interactions': defaultdict(list),
        'expressions': {}
    }
    
    with open(file_path, 'r') as f:
        # Skip header if present
        lines = [line.strip() for line in f.readlines() if line.strip()]
        if lines[0].startswith('targets') or lines[0].startswith('target'):
            lines = lines[1:]
        
        for line in lines:
            if not line or line.startswith('#'):
                continue
                
            # Split target and expression
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 2:
                continue
                
            target = parts[0]
            expression = ','.join(parts[1:])  # In case there are commas in the expression
            
            # Add to genes list if not already present
            if target not in network['genes']:
                network['genes'].append(target)
            
            # Parse interactions
            interacting_genes = set(re.findall(r'\b[A-Za-z][A-Za-z0-9_]*\b', expression))
            for gene in interacting_genes:
                if gene != target and gene not in network['interactions'][target]:
                    network['interactions'][target].append(gene)
            
            # Store expression
            network['expressions'][target] = expression
    
    return network


# Example usage matching the R function:
def generate_test_example_file():
    # Create the example file
    with open("example.bn", "w") as f:
        f.write("targets, factors\n")
        f.write("Gene1, Gene1\n")
        f.write("Gene2, Gene1 & Gene5 & !Gene4\n")
        f.write("Gene3, Gene3\n")
        f.write("Gene4, Gene3 & !(Gene1 & Gene5)\n")
        f.write("Gene5, !Gene2\n")
    
    # Load and print the network
    network = load_network("example.bn")
    
    # Print in a format similar to BoolNet's output
    print("Boolean network with", len(network['genes']), "genes")
    print("\nGenes:")
    print(", ".join(network['genes']))
    
    print("\nInteractions:")
    for gene in sorted(network['genes']):
        print(f"{gene}: {', '.join(network['interactions'][gene])}")
    
    print("\nExpressions:")
    for gene in sorted(network['genes']):
        print(f"{gene} = {network['expressions'][gene]}")
    
    return network


# Test it
if __name__ == "__main__":
    network = generate_test_example_file()