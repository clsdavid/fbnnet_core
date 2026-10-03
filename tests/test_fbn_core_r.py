import numpy as np

import unittest
import fbnnet_core  # This is your C++-wrapped Python module
import fbnnet_utils
import fbnnet_matrix
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


class TesteProbabilities(unittest.TestCase):
       
    def test_getGeneProbabilities_basic_CycD_p27(self):
        # Prepare input data
        main_params = setupdata()
        print(main_params["testseries"])


        target_gene = ["CycD"]
        conditional_genes = ["p27"]
        temporal = 1

        # Call the function
        probability = fbnnet_core.getGeneProbabilities_basic(main_params, None, target_gene, conditional_genes, temporal)
        probability = probability["1"]
        # # Access results


        # note conditional on top and target on bottom
        # computation_Matrix cond - target
        # when count, please disregard 9
        # p27:    1 1 1 0 9 1 1 1 0 9 0 1 1 
        # CycD:   0 1 0 9 1 0 0 1 9 1 1 1 1 

        # computation_Matrix_c target - cond
        # CycD: 1 0 1 0 9 1 0 0 1 9 1 1 1 
        # p27:  1 1 0 9 1 1 1 0 9 0 1 1 1 

        self.assertTrue(probability["target_T_count"] == 5)
        self.assertTrue(probability["target_F_count"] == 4)
        self.assertTrue(probability["count_cond_T_target_T"] == 4)
        self.assertTrue(probability["count_cond_F_target_T"] == 1)
        self.assertTrue(probability["count_cond_T_target_F"] == 4)
        self.assertTrue(probability["count_cond_F_target_F"] == 0)
        self.assertTrue(probability["cond_T_count"] == 8)
        self.assertTrue(probability["cond_F_count"] == 1)
        self.assertTrue(probability["count_cond_T_target_T_c"] == 5)
        self.assertTrue(probability["count_cond_F_target_T_c"] == 2)
        self.assertTrue(probability["count_cond_T_target_F_c"] == 1)
        self.assertTrue(probability["count_cond_F_target_F_c"] == 1)
        self.assertTrue(probability["cond_T_count_c"] == 6)
        self.assertTrue(probability["cond_F_count_c"] == 3)

    def test_getGeneProbabilities_basic_p27_CycD(self):
        # Prepare input data
        main_params = setupdata()

        target_gene = ["p27"]
        conditional_genes = ["CycD"]
        temporal = 1

        # Call the function
        probability = fbnnet_core.getGeneProbabilities_basic(main_params, None, target_gene, conditional_genes, temporal)
        probability = probability["1"]
        # # Access results

        # print("Time step:", probability["timestep"])
        # print("target_T_count:", probability["target_T_count"])
        # print("target_F_count:", probability["target_F_count"])
        # print("cond_T_count:", probability["cond_T_count"])
        # print("cond_F_count:", probability["cond_F_count"])
        # print("cond_T_count_c:", probability["cond_T_count_c"])
        # print("cond_F_count_c:", probability["cond_F_count_c"])
        # print("count_cond_T_target_T:", probability["count_cond_T_target_T"])
        # print("count_cond_F_target_T:", probability["count_cond_F_target_T"])
        # print("count_cond_T_target_F:", probability["count_cond_T_target_F"])
        # print("count_cond_F_target_F:", probability["count_cond_F_target_F"])
        # print("count_target_T_cond_T:", probability["count_target_T_cond_T"])
        # print("count_target_F_cond_T:", probability["count_target_F_cond_T"])
        # print("count_target_T_cond_F:", probability["count_target_T_cond_F"])
        # print("count_target_F_cond_F:", probability["count_target_F_cond_F"])

        # note conditional on top and target on bottom
        # computation_Matrix cond - target
        # when count, please disregard 9
        # CycD: 1 0 1 0 9 1 0 0 1 9 1 1 1 
        # p27:  1 1 0 9 1 1 1 0 9 0 1 1 1 
        # computation_Matrix_c target - cond
        # p27:  1 1 1 0 9 1 1 1 0 9 0 1 1 
        # CycD: 0 1 0 9 1 0 0 1 9 1 1 1 1 

        self.assertTrue(probability["target_T_count"] == 7)
        self.assertTrue(probability["target_F_count"] == 2)
        self.assertTrue(probability["count_cond_T_target_T"] == 5)
        self.assertTrue(probability["count_cond_F_target_T"] == 2)
        self.assertTrue(probability["count_cond_T_target_F"] == 1)
        self.assertTrue(probability["count_cond_F_target_F"] == 1)
        self.assertTrue(probability["cond_T_count"] == 6)
        self.assertTrue(probability["cond_F_count"] == 3)
        self.assertTrue(probability["count_cond_T_target_T_c"] == 4)
        self.assertTrue(probability["count_cond_F_target_T_c"] == 1)
        self.assertTrue(probability["count_cond_T_target_F_c"] == 4)
        self.assertTrue(probability["count_cond_F_target_F_c"] == 0)
        self.assertTrue(probability["cond_T_count_c"] == 8)
        self.assertTrue(probability["cond_F_count_c"] == 1)


    def test_getGeneProbabilities_basic_p27_CycE(self):
        # Prepare input data
        main_params = setupdata()

        target_gene = ["p27"]
        conditional_genes = ["CycE"]
        temporal = 1

        # Call the function
        probability = fbnnet_core.getGeneProbabilities_basic(main_params, None, target_gene, conditional_genes, temporal)
        probability = probability["1"]
        # # Access results


        # note conditional on top and target on bottom
        # computation_Matrix cond - target
        # when count, please disregard 9
        # CycE: 1 0 1 0 9 0 0 0 0 9 0 0 0 
        # p27:  1 1 0 9 1 1 1 0 9 0 1 1 1 

        # computation_Matrix_c target - cond
        # p27:  1 1 1 0 9 1 1 1 0 9 0 1 1 
        # CycE: 0 1 0 9 0 0 0 0 9 0 0 0 0 

        self.assertTrue(probability["target_T_count"] == 7)
        self.assertTrue(probability["target_F_count"] == 2)
        self.assertTrue(probability["count_cond_T_target_T"] == 1)
        self.assertTrue(probability["count_cond_F_target_T"] == 6)
        self.assertTrue(probability["count_cond_T_target_F"] == 1)
        self.assertTrue(probability["count_cond_F_target_F"] == 1)
        self.assertTrue(probability["cond_T_count"] == 2)
        self.assertTrue(probability["cond_F_count"] == 7)
        self.assertTrue(probability["count_cond_T_target_T_c"] == 1)
        self.assertTrue(probability["count_cond_F_target_T_c"] == 0)
        self.assertTrue(probability["count_cond_T_target_F_c"] == 7)
        self.assertTrue(probability["count_cond_F_target_F_c"] == 1)
        self.assertTrue(probability["cond_T_count_c"] == 8)
        self.assertTrue(probability["cond_F_count_c"] == 1)

    def test_getGeneProbabilities_basic_CycE_E2F(self):
        # Prepare input data
        main_params = setupdata()

        print(main_params["testseries"])
        target_gene = ["CycE"]
        conditional_genes = ["E2F"]
        temporal = 1

        fixedgenestate = {
            "p27": 1,
            "CycD": 0
        }
  
        # Call the function
        probability = fbnnet_core.getGeneProbabilities_basic(main_params, fixedgenestate, target_gene, conditional_genes, temporal)
        probability = probability["1"]
        # # Access results




        # note conditional on top and target on bottom
        # computation_Matrix cond - target, where fixed genes "CycD": 0, "p27": 1
        # when count, please disregard 9
        # CycD: 1 0 1 0 9 1 0 0 1 9 1 1 1 
        # p27:  1 1 1 0 9 1 1 1 0 9 0 1 1 
        # E2F:  1 1 1 1 9 1 1 1 1 9 1 1 1 
        # CycE: 0 1 0 9 0 0 0 0 9 0 0 0 0 

        # computation_Matrix_c target - cond
        # CycE: 1 0 1 0 9 0 0 0 0 9 0 0 0 
        # CycD: 0 1 0 9 1 0 0 1 9 1 1 1 1 
        # p27:  1 1 0 9 1 1 1 0 9 0 1 1 1 
        # E2F:  1 1 1 9 1 1 1 1 9 1 1 1 1 

        self.assertTrue(probability["target_T_count"] == 1)
        self.assertTrue(probability["target_F_count"] == 8)
        self.assertTrue(probability["count_cond_T_target_T"] == 1)
        self.assertTrue(probability["count_cond_F_target_T"] == 0)
        self.assertTrue(probability["count_cond_T_target_F"] == 2)
        self.assertTrue(probability["count_cond_F_target_F"] == 0)
        self.assertTrue(probability["cond_T_count"] == 3)
        self.assertTrue(probability["cond_F_count"] == 0)
        self.assertTrue(probability["count_cond_T_target_T_c"] == 1)
        self.assertTrue(probability["count_cond_F_target_T_c"] == 2)
        self.assertTrue(probability["count_cond_T_target_F_c"] == 0)
        self.assertTrue(probability["count_cond_F_target_F_c"] == 0)
        self.assertTrue(probability["cond_T_count_c"] == 1)
        self.assertTrue(probability["cond_F_count_c"] == 2)

class TesteProbabilitiesAdvanced(unittest.TestCase):
       
    def test_getGeneProbabilities_Advanced_CycD_p27(self):
        # Prepare input data
        main_params = setupdata()
        print(main_params["testseries"])


        target_gene = ["CycD"]
        conditional_genes = ["p27"]
        temporal = 1

        # Call the function
        basic_measures = fbnnet_core.getGeneProbabilities_basic(main_params, None, target_gene, conditional_genes, temporal)
        # # Access results
        probability = fbnnet_core.getGeneProbabilities_advanced(basic_measures)
        probability = probability["getBestFitP"]

        self.assertTrue(probability["TT"] == 0.5)
        self.assertTrue(probability["FT"] == 0.5)
        self.assertTrue(probability["TF"] == 1.0)
        self.assertTrue(probability["FF"] == 0)
        
        # test counter
        self.assertTrue(probability["TT_c"] == 0.83333)
        self.assertTrue(probability["FT_c"] == 0.16667)
        self.assertTrue(probability["TF_c"] == 0.66667)
        self.assertTrue(probability["FF_c"] == 0.33333)

 
    def test_getGeneProbabilities_Advanced_p27_CycD(self):
        # Prepare input data
        main_params = setupdata()
        print(main_params["testseries"])


        target_gene = ["p27"]
        conditional_genes = ["CycD"]
        temporal = 1

        # Call the function
        basic_measures = fbnnet_core.getGeneProbabilities_basic(main_params, None, target_gene, conditional_genes, temporal)
        # # Access results
        probability = fbnnet_core.getGeneProbabilities_advanced(basic_measures)
        probability = probability["getBestFitP"]

        self.assertTrue(probability["TT"] == 0.83333)
        self.assertTrue(probability["FT"] == 0.16667)
        self.assertTrue(probability["TF"] == 0.66667)
        self.assertTrue(probability["FF"] == 0.33333)

        
        # test counter
        self
        self.assertTrue(probability["TT_c"] == 0.5)
        self.assertTrue(probability["FT_c"] == 0.5)
        self.assertTrue(probability["TF_c"] == 1.0)
        self.assertTrue(probability["FF_c"] == 0)

    def test_getGeneProbabilities_Advanced_p27_CycE(self):
        # Prepare input data
        main_params = setupdata()
        print(main_params["testseries"])


        target_gene = ["p27"]
        conditional_genes = ["CycE"]
        temporal = 1

        # Call the function
        basic_measures = fbnnet_core.getGeneProbabilities_basic(main_params, None, target_gene, conditional_genes, temporal)
        # # Access results
        probability = fbnnet_core.getGeneProbabilities_advanced(basic_measures)
        probability = probability["getBestFitP"]

        self.assertTrue(probability["TT"] == 0.5)
        self.assertTrue(probability["FT"] == 0.5)
        self.assertTrue(probability["TF"] == 0.85714)
        self.assertTrue(probability["FF"] == 0.14286)

        
        # test counter
        self.assertTrue(probability["TT_c"] == 0.125)
        self.assertTrue(probability["FT_c"] == 0.875)
        self.assertTrue(probability["TF_c"] == 0)
        self.assertTrue(probability["FF_c"] == 1)


    def test_getGeneProbabilities_Advanced_CycE_E2F(self):
        # Prepare input data
        main_params = setupdata()
        print(main_params["testseries"])

        target_gene = ["CycE"]
        conditional_genes = ["E2F"]
        temporal = 1

        fixedgenestate = {
            "p27": 1,
            "CycD": 0
        }
  
        # Call the function
        basic_measures = fbnnet_core.getGeneProbabilities_basic(main_params, fixedgenestate, target_gene, conditional_genes, temporal)
        # # Access results
        probability = fbnnet_core.getGeneProbabilities_advanced(basic_measures)
        probability = probability["getBestFitP"]

        self.assertTrue(probability["TT"] == 0.33333)
        self.assertTrue(probability["FT"] == 0.66667)
        self.assertTrue(probability["TF"] == 0)
        self.assertTrue(probability["FF"] == 0)

        
        # test counter
        self.assertTrue(probability["TT_c"] == 1)
        self.assertTrue(probability["FT_c"] == 0)
        self.assertTrue(probability["TF_c"] == 1)
        self.assertTrue(probability["FF_c"] == 0)

class TestGetProbabilities(unittest.TestCase):
    def test_getGeneProbabilities_CycD_p27(self):
        # Prepare input data
        main_params = setupdata()
        print(main_params["testseries"])

        target_gene = ["CycD"]
        conditional_genes = ["p27"]
        temporal = 1

        # Call the function
        probability = fbnnet_core.getGeneProbabilities(main_params, None, target_gene, conditional_genes, temporal)
        # # Access results
        
        getBestFitP = probability["BestFitP"]
        getBestFitN = probability["BestFitN"]


        self.assertTrue(getBestFitP["TT"] == 0.5)
        self.assertTrue(getBestFitP["FT"] == 0.5)
        self.assertTrue(getBestFitP["TF"] == 1.0)
        self.assertTrue(getBestFitP["FF"] == 0)
        
        # test counter
        self.assertTrue(getBestFitP["TT_c"] == 0.83333)
        self.assertTrue(getBestFitP["FT_c"] == 0.16667)
        self.assertTrue(getBestFitP["TF_c"] == 0.66667)
        self.assertTrue(getBestFitP["FF_c"] == 0.33333)

    def test_getGeneProbabilities_Advanced_p27_CycD(self):
        # Prepare input data
        main_params = setupdata()
        print(main_params["testseries"])


        target_gene = ["p27"]
        conditional_genes = ["CycD"]
        temporal = 1

        # Call the function
        probability = fbnnet_core.getGeneProbabilities(main_params, None, target_gene, conditional_genes, temporal)
        
        getBestFitP = probability["BestFitP"]
        getBestFitN = probability["BestFitN"]

        self.assertTrue(getBestFitP["TT"] == 0.83333)
        self.assertTrue(getBestFitP["FT"] == 0.16667)
        self.assertTrue(getBestFitP["TF"] == 0.66667)
        self.assertTrue(getBestFitP["FF"] == 0.33333)

        
        # test counter
        self
        self.assertTrue(getBestFitP["TT_c"] == 0.5)
        self.assertTrue(getBestFitP["FT_c"] == 0.5)
        self.assertTrue(getBestFitP["TF_c"] == 1.0)
        self.assertTrue(getBestFitP["FF_c"] == 0)

    def test_getGeneProbabilities_Advanced_p27_CycE(self):
        # Prepare input data
        main_params = setupdata()
        print(main_params["testseries"])


        target_gene = ["p27"]
        conditional_genes = ["CycE"]
        temporal = 1

        # Call the function
        probability = fbnnet_core.getGeneProbabilities(main_params, None, target_gene, conditional_genes, temporal)
        # # Access results
        
        getBestFitP = probability["BestFitP"]
        getBestFitN = probability["BestFitN"]

        self.assertTrue(getBestFitP["TT"] == 0.5)
        self.assertTrue(getBestFitP["FT"] == 0.5)
        self.assertTrue(getBestFitP["TF"] == 0.85714)
        self.assertTrue(getBestFitP["FF"] == 0.14286)

        
        # test counter
        self.assertTrue(getBestFitP["TT_c"] == 0.125)
        self.assertTrue(getBestFitP["FT_c"] == 0.875)
        self.assertTrue(getBestFitP["TF_c"] == 0)
        self.assertTrue(getBestFitP["FF_c"] == 1)


    def test_getGeneProbabilities_Advanced_CycE_E2F(self):
        # Prepare input data
        main_params = setupdata()
        print(main_params["testseries"])

        target_gene = ["CycE"]
        conditional_genes = ["E2F"]
        temporal = 1

        fixedgenestate = {
            "p27": 1,
            "CycD": 0
        }
  
        # Call the function
        probability = fbnnet_core.getGeneProbabilities(main_params, fixedgenestate, target_gene, conditional_genes, temporal)
        # # Access results
        
        getBestFitP = probability["BestFitP"]
        getBestFitN = probability["BestFitN"]

        self.assertTrue(getBestFitP["TT"] == 0.33333)
        self.assertTrue(getBestFitP["FT"] == 0.66667)
        self.assertTrue(getBestFitP["TF"] == 0)
        self.assertTrue(getBestFitP["FF"] == 0)

        
        # test counter
        self.assertTrue(getBestFitP["TT_c"] == 1)
        self.assertTrue(getBestFitP["FT_c"] == 0)
        self.assertTrue(getBestFitP["TF_c"] == 1)
        self.assertTrue(getBestFitP["FF_c"] == 0)

class TestGetProbabilitiesTemporal(unittest.TestCase):
    def test_getGeneProbabilities_CycD_p27(self):
        # Prepare input data
        main_params = setupdata()
        print(main_params["testseries"])

        target_gene = ["CycD"]
        conditional_genes = ["p27"]
        temporal = 2

        # Call the function
        probability = fbnnet_core.getGeneProbabilities(main_params, None, target_gene, conditional_genes, temporal)
        # # Access results
        
        getBestFitP = probability["BestFitP"]
        getBestFitN = probability["BestFitN"]

        self.assertTrue(getBestFitN["timestep"] == 1)
        self.assertTrue(getBestFitP["TT"] == 0.5)
        self.assertTrue(getBestFitP["FT"] == 0.5)
        self.assertTrue(getBestFitP["TF"] == 1.0)
        self.assertTrue(getBestFitP["FF"] == 0)
        
        # test counter
        self.assertTrue(getBestFitP["TT_c"] == 0.83333)
        self.assertTrue(getBestFitP["FT_c"] == 0.16667)
        self.assertTrue(getBestFitP["TF_c"] == 0.66667)
        self.assertTrue(getBestFitP["FF_c"] == 0.33333)


    def test_getGeneProbabilities_Advanced_p27_CycD(self):
        # Prepare input data
        main_params = setupdata()
        print(main_params["testseries"])


        target_gene = ["p27"]
        conditional_genes = ["CycD"]
        temporal = 2

        # Call the function
        probability = fbnnet_core.getGeneProbabilities(main_params, None, target_gene, conditional_genes, temporal)
        
        getBestFitP = probability["BestFitP"]
        getBestFitN = probability["BestFitN"]
        self.assertTrue(getBestFitN["timestep"] == 2)
        self.assertTrue(getBestFitP["TT"] == 1)
        self.assertTrue(getBestFitP["FT"] == 0)
        self.assertTrue(getBestFitP["TF"] == 0)
        self.assertTrue(getBestFitP["FF"] == 1)

        # test counter
        self.assertTrue(getBestFitP["TT_c"] == 0.6)
        self.assertTrue(getBestFitP["FT_c"] == 0.4)
        self.assertTrue(getBestFitP["TF_c"] == 1.0)
        self.assertTrue(getBestFitP["FF_c"] == 0)

    def test_getGeneProbabilities_Advanced_p27_CycE(self):
        # Prepare input data
        main_params = setupdata()
        print(main_params["testseries"])


        target_gene = ["p27"]
        conditional_genes = ["CycE"]
        temporal = 2

        # Call the function
        probability = fbnnet_core.getGeneProbabilities(main_params, None, target_gene, conditional_genes, temporal)
        # # Access results
        
        getBestFitP = probability["BestFitP"]
        getBestFitN = probability["BestFitN"]
        self.assertTrue(getBestFitN["timestep"] == 1)
        self.assertTrue(getBestFitN["TT"] == 0.5)
        self.assertTrue(getBestFitN["FT"] == 0.5)
        self.assertTrue(getBestFitN["TF"] == 0.85714)
        self.assertTrue(getBestFitN["FF"] == 0.14286)

        
        # test counter
        self.assertTrue(getBestFitN["TT_c"] == 0.125)
        self.assertTrue(getBestFitN["FT_c"] == 0.875)
        self.assertTrue(getBestFitN["TF_c"] == 0)
        self.assertTrue(getBestFitN["FF_c"] == 1)


    def test_getGeneProbabilities_Advanced_CycE_E2F(self):
        # Prepare input data
        main_params = setupdata()
        print(main_params["testseries"])

        target_gene = ["CycE"]
        conditional_genes = ["E2F"]
        temporal = 2

        fixedgenestate = {
            "p27": 1,
            "CycD": 0
        }
  
        # Call the function
        probability = fbnnet_core.getGeneProbabilities(main_params, fixedgenestate, target_gene, conditional_genes, temporal)
        # # Access results
        
        getBestFitP = probability["BestFitP"]
        getBestFitN = probability["BestFitN"]

        self.assertTrue(getBestFitP["TT"] == 0)
        self.assertTrue(getBestFitP["FT"] == 1)
        self.assertTrue(getBestFitP["TF"] == 0)
        self.assertTrue(getBestFitP["FF"] == 0)
        self.assertTrue(getBestFitN["timestep"] == 2)

        
        # test counter
        self.assertTrue(getBestFitP["TT_c"] == 0)
        self.assertTrue(getBestFitP["FT_c"] == 0)
        self.assertTrue(getBestFitP["TF_c"] == 1)
        self.assertTrue(getBestFitP["FF_c"] == 0)


class TestGetProbabilitiesBasicMeasuresRParity(unittest.TestCase):
    """
    The `basic_measures` count assertions of R's test-getgeneprababilities.R and
    test-getgeneprobabilitiestemporal.R (R names lenTT/lenTF/lenFT/lenFF map to
    count_cond_<cond>_target_<target> here).
    """

    # (target, condition, fixed state, temporal, best fit, counts in R's order:
    #  target_T, target_F, lenTT, lenTF, lenFT, lenFF, cond_T, cond_F)
    CASES = {
        "getgeneprababilities CycD<-p27 t1": ("CycD", "p27", None, 1, "BestFitP", (5, 4, 4, 1, 4, 0, 8, 1)),
        "getgeneprababilities CycE<-E2F fixed t1": ("CycE", "E2F", {"p27": 1, "CycD": 0}, 1, "BestFitP", (1, 8, 1, 0, 2, 0, 3, 0)),
        "temporal CycD<-p27 t2": ("CycD", "p27", None, 2, "BestFitP", (5, 4, 4, 1, 4, 0, 8, 1)),
        "temporal p27<-CycD t2": ("p27", "CycD", None, 2, "BestFitP", (4, 2, 4, 0, 0, 2, 4, 2)),
        "temporal p27<-CycE t2 (N)": ("p27", "CycE", None, 2, "BestFitN", (7, 2, 1, 6, 1, 1, 2, 7)),
        "temporal CycE<-E2F fixed t2": ("CycE", "E2F", {"p27": 1, "CycD": 0}, 2, "BestFitP", (1, 5, 0, 0, 2, 0, 2, 0)),
    }
    KEYS = (
        "target_T_count", "target_F_count",
        "count_cond_T_target_T", "count_cond_F_target_T", "count_cond_T_target_F", "count_cond_F_target_F",
        "cond_T_count", "cond_F_count",
    )

    def test_basic_measures_counts_match_r(self):
        for name, (target, cond, fixed, temporal, best_fit, expected) in self.CASES.items():
            with self.subTest(case=name):
                result = fbnnet_core.getGeneProbabilities(
                    setupdata(), fixed, [target], [cond], temporal, True
                )
                basic = result[best_fit]["basic_measures"]
                self.assertEqual(tuple(basic[key] for key in self.KEYS), expected)


if __name__ == "__main__":

    unittest.main()

        # self.assertEqual(fbnnet_core.add(-1, 1), 0)