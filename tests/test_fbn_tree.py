import numpy as np

import unittest
import fbnnet_core  # This is your C++-wrapped Python module
import fbnnet_utils
import fbnnet_matrix
import fbnnet_tree
from types import SimpleNamespace


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

    


if __name__ == "__main__":

    unittest.main()

        # self.assertEqual(fbnnet_core.add(-1, 1), 0)