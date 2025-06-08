import numpy as np

import unittest
import fbnnet_core  # This is your C++-wrapped Python module
import fbnnet_utils
import fbnnet_matrix
import fbnnet_tree
from types import SimpleNamespace
import pandas as pd

def setupdata():
    #' Create an Orchard cube
#'
#' This is the main function(s) to generate a single Orchard Cube or a group
#' of cubes
#'
#' @param target_genes A vector of genes that will be treated as target genes
#' @param conditional_genes All genes that are available for building up the
#'  cube
#' @param timeseriesCube A list of samples in which a sample is a matrix that
#'  contains gene states 
#'  where genes in rows and time points in columns.
#' @param maxK The maximum level the cube can dig in
#' @param temporal A value that used to be 1 indicates the previous steps the
#'  current one can depend on
#' @param useParallel If it is TRUE, the constructing will run it in parallel,
#'  otherwise in a single thread

    genes_input = ["CycD", "p27", "CycE", "E2F"]
    #     testseries <- list()
    # testseries[[1]] <- matrix(c(1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1), nrow = 4, ncol = 4, byrow = FALSE, dimnames <- list(genesInput, c("1", "2", "3", 
    #     "4")))
    # testseries[[2]] <- matrix(c(1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1), nrow = 4, ncol = 4, byrow = FALSE, dimnames <- list(genesInput, c("1", "2", "3", 
    #     "4")))
    # testseries[[3]] <- matrix(c(1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1), nrow = 4, ncol = 4, byrow = FALSE, dimnames <- list(genesInput, c("1", "2", "3", 
    #     "4")))
    
    # Create test series matrices
    testseries = [
        np.array([
            [1, 0, 1, 0],
            [1, 1, 1, 0],
            [1, 0, 1, 0],
            [1, 1, 1, 1]
        ], dtype=np.float64),
        
        np.array([
            [1, 0, 0, 1],
            [1, 1, 1, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 1]
        ], dtype=np.float64),
        
        np.array([
            [1, 1, 1, 1],
            [0, 1, 1, 1],
            [0, 0, 0, 0],
            [1, 1, 1, 1]
        ], dtype=np.float64)
    ]


    # Convert numpy arrays to FBNMatrix objects
    for i, mat in enumerate(testseries):
        testseries[i] = fbnnet_matrix.FBNMatrix(mat, genes_input, [str(j+1) for j in range(mat.shape[1])])

    # Initialize containers
    getCurrentStates = []
    getPreviousStates = []
    getCurrentStates_c = []
    getPreviousStates_c = []
    
    # Process matrices (note: we're skipping the extractGeneStateFromTimeSeriesCube calls)
    for index in range(3):
        # In R this would be: 1-3 but in Python we use 0-2
        # In Python we'll just store the original matrices since we can't call the R function
        # In a real implementation, you would call your Python equivalent here
        # initializing all states and combined all sample's states with 9s.
        temporal = index + 1
        getCurrentStates.append(fbnnet_core.extract_gene_state_from_time_series_cube(testseries, temporal))
        getPreviousStates.append(getCurrentStates[index])
        getCurrentStates_c.append(getCurrentStates[index])
        getPreviousStates_c.append(getCurrentStates[index])
    
    # Calculate totals
    total_timepoints = sum(mat.matrix_t().shape[1] for mat in testseries)
    total_samples = len(testseries)
    all_gene_names = genes_input
    
    # Create namespace object similar to R's environment
    main_parameters = {
        "currentStates": getCurrentStates,
        "previousStates": getPreviousStates,
        "currentStates_c": getCurrentStates_c,
        "previousStates_c": getPreviousStates_c,
        "total_samples": total_samples,
        "rownames": all_gene_names,
        "total_timepoints": total_timepoints,
        "testseries": testseries
    }

    return main_parameters


def generate_test_example():
    from py.boolnet import load_network
    from py.data_utils import generateAllCombinationBinary, generateBoolNetTimeseries
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

    getCurrentStates = []
    getPreviousStates = []
    getCurrentStates_c = []
    getPreviousStates_c = []
    
        # Convert numpy arrays to FBNMatrix objects
    for i, mat in enumerate(trainingseries):
        trainingseries[i] = fbnnet_matrix.FBNMatrix(mat, network["genes"], [str(j+1) for j in range(mat.shape[1])])

    # Process matrices (note: we're skipping the extractGeneStateFromTimeSeriesCube calls)
    for index in range(3):
        # In R this would be: 1-3 but in Python we use 0-2
        # In Python we'll just store the original matrices since we can't call the R function
        # In a real implementation, you would call your Python equivalent here
        # initializing all states and combined all sample's states with 9s.
        temporal = index + 1
        getCurrentStates.append(fbnnet_core.extract_gene_state_from_time_series_cube(trainingseries, temporal))
        getPreviousStates.append(getCurrentStates[index])
        getCurrentStates_c.append(getCurrentStates[index])
        getPreviousStates_c.append(getCurrentStates[index])
    
    # Calculate totals
    total_timepoints = sum(mat.matrix_t().shape[1] for mat in trainingseries)
    total_samples = len(trainingseries)
    all_gene_names = network["genes"]
    
    # Create namespace object similar to R's environment
    main_parameters = {
        "currentStates": getCurrentStates,
        "previousStates": getPreviousStates,
        "currentStates_c": getCurrentStates_c,
        "previousStates_c": getPreviousStates_c,
        "total_samples": total_samples,
        "rownames": all_gene_names,
        "total_timepoints": total_timepoints,
        "testseries": trainingseries
    }

    return main_parameters

class TestFBNTree(unittest.TestCase):
    def test_fbn_tree_CycD(self):
        # Prepare input data
        main_params = setupdata()
        print(main_params["testseries"])

        target_gene = ["CycD"]
        temporal = 3
        genes = main_params["rownames"]
        # convert genes to vector list
        # genes = [str(gene) for gene in genes]
        # Call the function
        measurements = fbnnet_tree.getGeneProbabilities_measurements(["CycD"], main_params, genes, None, temporal, False)
        # # Access results
        
        print(measurements)
    def test_fbn_tree_p27(self):
        # Prepare input data
        main_params = setupdata()
        print(main_params["testseries"])

        target_gene = ["CycD"]
        temporal = 3
        genes = main_params["rownames"]
        # convert genes to vector list
        # genes = [str(gene) for gene in genes]
        # Call the function
        measurements = fbnnet_tree.getGeneProbabilities_measurements(["CycD"], main_params, genes, None, temporal, False)
        # # Access results
        
        print(measurements)

    
class TestFBNTreeBuild(unittest.TestCase):

    def test_filterTargetGenesByConditionGenes(self):
        # Prepare input data
        from py.boolnet import load_network
        from py.data_utils import generateAllCombinationBinary, generateBoolNetTimeseries
        from py.cube import convert_df_main_parameters
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
        main_params = convert_df_main_parameters(trainingseries, 1)

        # Call the function
        filtered_genes = fbnnet_tree.filterTargetGenesByConditionGenes(genes, main_params, genes, None, 1)
        # Access results
        self.assertTrue(filtered_genes == genes)  # In this case, all genes should be retained
        print("Filtered genes: ", filtered_genes)

    def test_process_cube_algorithm(self):
        # Prepare input data
        main_params = generate_test_example()
        # print(main_params["testseries"])

        genes = main_params["rownames"]
        for gene in genes:
            cube = fbnnet_tree.process_cube_algorithm(gene, genes, 4, 1, main_params, None, None)
            print("cube for gene: ", gene)
            print(cube)

    def test_fbn_tree_build(self):
        # Prepare input data
        main_params = generate_test_example()
        # print(main_params["testseries"])

        genes = main_params["rownames"]
        for gene in genes:
            cube = fbnnet_tree.buildProbabilityTreeOnTargetGene([gene], main_params, genes, None, None, 4, 1)
            print("cube for gene: ", gene)
            print(cube)

    def test_fbn_tree_individual(self):
        # Prepare input data
        main_params = setupdata()

        genesInput = ["CycD", "p27", "CycE", "E2F"]
        # # Access results
        cube = fbnnet_tree.buildProbabilityTreeOnTargetGene(["CycD"], main_params, genesInput, None, None, 4, 1)

        cube = fbnnet_tree.buildProbabilityTreeOnTargetGene(["CycD"], main_params, genesInput, None, None, 4, 2)

        cube = fbnnet_tree.buildProbabilityTreeOnTargetGene(["CycD"], main_params, genesInput, None, None, 4, 3)

        cube = fbnnet_tree.buildProbabilityTreeOnTargetGene(["p27"], main_params, genesInput, None, None, 4, 1)

        cube = fbnnet_tree.buildProbabilityTreeOnTargetGene(["p27"], main_params, genesInput, None, None, 4, 2)

        cube = fbnnet_tree.buildProbabilityTreeOnTargetGene(["p27"], main_params, genesInput, None, None, 4, 3)

        cube = fbnnet_tree.buildProbabilityTreeOnTargetGene(["CycE"], main_params, genesInput, None, None, 4, 1)

        cube = fbnnet_tree.buildProbabilityTreeOnTargetGene(["CycE"], main_params, genesInput, None, None, 4, 2)

        cube = fbnnet_tree.buildProbabilityTreeOnTargetGene(["CycE"], main_params, genesInput, None, None, 4, 3)

        cube = fbnnet_tree.buildProbabilityTreeOnTargetGene(["E2F"], main_params, genesInput, None, None, 4, 1)

        cube = fbnnet_tree.buildProbabilityTreeOnTargetGene(["E2F"], main_params, genesInput, None, None, 4, 2)

        cube = fbnnet_tree.buildProbabilityTreeOnTargetGene(["E2F"], main_params, genesInput, None, None, 4, 3)

        print(cube)

if __name__ == "__main__":

    unittest.main()
    network = generate_test_example()
    print(network)
        # self.assertEqual(fbnnet_core.add(-1, 1), 0)