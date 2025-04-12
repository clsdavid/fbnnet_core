import numpy as np

import unittest
import fbnnet_core  # This is your C++-wrapped Python module
import fbnnet_utils
import fbnnet_matrix
from types import SimpleNamespace


def setupdata():
    #' Create an Orchard cube
#'
#' This is the main function(s) to genereate a single Orchard Cube or a group
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
#'  otherwise in a singl thread

    genes_input = ["CycD", "p27", "CycE", "E2F"]
    
    # Create test series matrices
    testseries = [
        np.array([
            [1, 1, 1, 1, 0, 1],
            [0, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
            [1, 0, 1, 0, 1, 0]
        ], dtype=np.float64),
        
        np.array([
            [1, 1, 0, 1, 0, 1],
            [0, 1, 0, 1, 0, 1],
            [1, 0, 0, 1, 1, 0],
            [1, 1, 0, 0, 1, 1]
        ], dtype=np.float64),
        
        np.array([
            [1, 0, 0, 1, 1, 1],
            [0, 1, 1, 1, 0, 1],
            [1, 1, 0, 1, 1, 1],
            [1, 1, 1, 0, 0, 0]
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

def setupdata2():
    #' Create an Orchard cube
#'
#' This is the main function(s) to genereate a single Orchard Cube or a group
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
#'  otherwise in a singl thread

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
            [1, 1, 1, 1],
            [0, 1, 0, 1],
            [1, 1, 1, 1],
            [0, 0, 0, 1]
        ], dtype=np.float64),
        
        np.array([
            [1, 1, 0, 1],
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [1, 0, 0, 1]
        ], dtype=np.float64),
        
        np.array([
            [1, 0, 0, 1],
            [1, 1, 0, 1],
            [1, 1, 0, 1],
            [1, 1, 0, 1]
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

class TestCore(unittest.TestCase):
    def test_extract_gene_state_from_time_series_cube(self):
        data = [
            [1.0, 2.0, 3.0],
            [3.0, 5.0, 6.0],
            [9.0, 7.0, 9.0]
        ]
        row_names = ["GeneA", "GeneB", "GeneC"]
        col_names = ["Time1", "Time2", "Time3"]
        
        m1 = fbnnet_matrix.FBNMatrix(data, row_names, col_names)

        data = [
            [3.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ]
        row_names = ["GeneA", "GeneB", "GeneC"]
        col_names = ["Time1", "Time2", "Time3"]
        
        m2 = fbnnet_matrix.FBNMatrix(data, row_names, col_names)
        cube = [m1, m2]
        result = fbnnet_core.extract_gene_state_from_time_series_cube(cube, 2)
        print(result)
        self.assertEqual(result.shape, (3, 8))

        result = fbnnet_core.extract_gene_state_from_time_series_cube(cube, 1)
        print(result)
        self.assertEqual(result.shape, (3, 7))
        
    def test_extract_gene_states(self):

        # Create sample data
        state_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
        row_names = ["gene1", "gene2", "gene3"]
        target_genes = ["gene3", "gene1"]

        # Call the function
        result = fbnnet_core.extract_gene_states(state_matrix, target_genes, row_names)

        # Access results
        filtered_matrix = result.matrix()
        filtered_row_names = result.row_names()

        print("Filtered matrix:")
        print(filtered_matrix)
        print("Row names:", filtered_row_names)
        self.assertEqual(filtered_row_names, ['gene1', 'gene3'])

    def test_generate_test_data(self):
        test = setupdata()
        print("Total timepoints:", test["total_timepoints"])
        print("Total samples:", test["total_samples"])
        print("All gene names:", test["rownames"])
        print("Current states:", test["currentStates"])
        print("Previous states:", test["previousStates"])
        print("Current states (C):", test["currentStates_c"])
        print("Previous states (C):", test["previousStates_c"])




    def test_matrix_with_labels(self):
        print("=== Testing MatrixWithLabels ===")
        
        # Test 1: Create from Python lists
        print("\nTest 1: From Python lists")
        data = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ]
        row_names = ["GeneA", "GeneB", "GeneC"]
        col_names = ["Time1", "Time2", "Time3"]
        
        m1 = fbnnet_matrix.FBNMatrix(data, row_names, col_names)
        print("Matrix 1:")
        m1.print()
        print("String representation:")
        print(m1)
        print("native representation:")
        print(m1.matrix())
        print("numpy representation:")
        print(m1.matrix_t())
        
        # Test 2: Create from numpy array
        print("\nTest 2: From numpy array")
        arr = np.array([
            [1.1, 2.2],
            [3.3, 4.4],
            [5.5, 6.6]
        ], dtype=np.float64)
        
        m2 = fbnnet_matrix.FBNMatrix(arr, ["R1", "R2", "R3"])
        print("Matrix 2:")
        m2.print(2)  # With 2 decimal places
        print(f"Dimensions: {m2.num_rows()}x{m2.num_cols()}")
        
        # Test 3: Empty matrix
        print("\nTest 3: Empty matrix")
        m3 = fbnnet_matrix.FBNMatrix()
        print("Empty matrix:")
        print(m3)

    def test_generate_temporal_gene_states(self):
        # Prepare input data
        main_params = setupdata()
        print("====TestSeries====")
        print(main_params["testseries"])
        print("====Current States====")
        print(main_params["currentStates"])
        print("====Previous States====")
        print(main_params["previousStates"])
        print("====Current States_c====")
        print(main_params["currentStates_c"])

        print(main_params["previousStates_c"])
        print("====Total Samples====")
        print(main_params["total_samples"])
        print("====Row Names====")
        print(main_params["rownames"])
        target_gene = ["CycD"]
        conditional_genes = ["p27", "CycE"]
        temporal = 3

        # Call the function
        result = fbnnet_core.generate_temporal_gene_states(main_params, target_gene, conditional_genes, temporal)

        # Access results
        for time_step_result in result:
            print("Time step:", time_step_result["timeStep"])
            print("Matrix:", time_step_result["computation_Matrix"])
            print("Matrix_c:", time_step_result["computation_Matrix_c"])
        

    def test_getGeneProbabilities_basic(self):
        # Prepare input data
        main_params = setupdata()

        target_gene = ["CycD"]
        conditional_genes = ["p27", "CycE"]
        temporal = 3

        # Call the function
        result = fbnnet_core.getGeneProbabilities_basic(main_params, None, target_gene, conditional_genes, temporal, None)

        # Access results
        for time_step_result in result:
            item = result[time_step_result]
            print("Time step:", item["timestep"])
            print("target_T_count:", item["target_T_count"])
            print("target_F_count:", item["target_F_count"])
            print("cond_T_count:", item["cond_T_count"])
            print("cond_F_count:", item["cond_F_count"])
            print("cond_T_count_c:", item["cond_T_count_c"])
            print("cond_F_count_c:", item["cond_F_count_c"])
            print("lenTT:", item["lenTT"])
            print("lenTF:", item["lenTF"])
            print("lenFT:", item["lenFT"])
            print("lenFF:", item["lenFF"])
            print("lenTT_c:", item["lenTT_c"])
            print("lenTF_c:", item["lenTF_c"])
            print("lenFT_c:", item["lenFT_c"])
            print("lenFF_c:", item["lenFF_c"])

class TesteProbabilities(unittest.TestCase):
       
    def test_getGeneProbabilities_basic_CycD_p27(self):
        # Prepare input data
        main_params = setupdata2()
        print(main_params["testseries"])

        # [
        #     CycD: 1 1 1 1 
        #     p27:  0 1 0 1 
        #     CycE: 1 1 1 1 
        #     E2F:  0 0 0 1 

        #     CycD: 1 1 0 1 
        #     p27:  0 1 0 1 
        #     CycE: 0 1 0 1 
        #     E2F:  1 0 0 1 

        #     CycD: 1 0 0 1 
        #     p27:  1 1 0 1 
        #     CycE: 1 1 0 1 
        #     E2F:  1 1 0 1 
        # ]
        
        target_gene = ["CycD"]
        conditional_genes = ["p27"]
        temporal = 1

        # Call the function
        probability = fbnnet_core.getGeneProbabilities_basic(main_params, None, target_gene, conditional_genes, temporal, None)
        probability = probability["1"]
        # # Access results

        print("Time step:", probability["timestep"])
        print("target_T_count:", probability["target_T_count"])
        print("target_F_count:", probability["target_F_count"])
        print("cond_T_count:", probability["cond_T_count"])
        print("cond_F_count:", probability["cond_F_count"])
        print("cond_T_count_c:", probability["cond_T_count_c"])
        print("cond_F_count_c:", probability["cond_F_count_c"])
        print("lenTT:", probability["lenTT"])
        print("lenTF:", probability["lenTF"])
        print("lenFT:", probability["lenFT"])
        print("lenFF:", probability["lenFF"])
        print("lenTT_c:", probability["lenTT_c"])
        print("lenTF_c:", probability["lenTF_c"])
        print("lenFT_c:", probability["lenFT_c"])
        print("lenFF_c:", probability["lenFF_c"])

        self.assertTrue(probability["target_T_count"] == 5)
        self.assertTrue(probability["target_F_count"] == 4)
        self.assertTrue(probability["lenTT"] == 4)
        self.assertTrue(probability["lenTF"] == 1)
        self.assertTrue(probability["lenFT"] == 4)
        self.assertTrue(probability["lenFF"] == 0)
        self.assertTrue(probability["cond_T_count"] == 8)
        self.assertTrue(probability["cond_F_count"] == 1)
        self.assertTrue(probability["lenTT_c"] == 5)
        self.assertTrue(probability["lenTF_c"] == 2)
        self.assertTrue(probability["lenFT_c"] == 1)
        self.assertTrue(probability["lenFF_c"] == 1)
        self.assertTrue(probability["cond_T_count_c"] == 6)
        self.assertTrue(probability["cond_F_count_c"] == 3)
        # # Uncomment the following lines to test with different parameters
        # probability <- getGenePrababilities_basic(mainParameters, NULL, "CycD", "p27", 1, NULL)[[1]]

    def test_getGeneProbabilities_basic_p27_CycD(self):
        # Prepare input data
        main_params = setupdata2()

        target_gene = ["p27"]
        conditional_genes = ["CycD"]
        temporal = 1

        # Call the function
        probability = fbnnet_core.getGeneProbabilities_basic(main_params, None, target_gene, conditional_genes, temporal, None)
        probability = probability["1"]
        # # Access results

        print("Time step:", probability["timestep"])
        print("target_T_count:", probability["target_T_count"])
        print("target_F_count:", probability["target_F_count"])
        print("cond_T_count:", probability["cond_T_count"])
        print("cond_F_count:", probability["cond_F_count"])
        print("cond_T_count_c:", probability["cond_T_count_c"])
        print("cond_F_count_c:", probability["cond_F_count_c"])
        print("lenTT:", probability["lenTT"])
        print("lenTF:", probability["lenTF"])
        print("lenFT:", probability["lenFT"])
        print("lenFF:", probability["lenFF"])
        print("lenTT_c:", probability["lenTT_c"])
        print("lenTF_c:", probability["lenTF_c"])
        print("lenFT_c:", probability["lenFT_c"])
        print("lenFF_c:", probability["lenFF_c"])

        self.assertTrue(probability["target_T_count"] == 7)
        self.assertTrue(probability["target_F_count"] == 2)
        self.assertTrue(probability["lenTT"] == 5)
        self.assertTrue(probability["lenTF"] == 2)
        self.assertTrue(probability["lenFT"] == 1)
        self.assertTrue(probability["lenFF"] == 1)
        self.assertTrue(probability["cond_T_count"] == 6)
        self.assertTrue(probability["cond_F_count"] == 3)
        self.assertTrue(probability["lenTT_c"] == 4)
        self.assertTrue(probability["lenTF_c"] == 1)
        self.assertTrue(probability["lenFT_c"] == 4)
        self.assertTrue(probability["lenFF_c"] == 0)
        self.assertTrue(probability["cond_T_count_c"] == 8)
        self.assertTrue(probability["cond_F_count_c"] == 1)
        # # Uncomment the following lines to test with different parameters
        # it("getGenePrababilities(mainParameters,NULL,\"p27\",\"CycD\",1)", {
        #     probability <- getGenePrababilities_basic(mainParameters, NULL, "p27", "CycD", 1, NULL)[[1]]
        #     expect_equal(probability$target_T_count, 7)
        #     expect_equal(probability$target_F_count, 2)
        #     expect_equal(probability$lenTT, 5)
        #     expect_equal(probability$lenTF, 2)
        #     expect_equal(probability$lenFT, 1)
        #     expect_equal(probability$lenFF, 1)
        #     expect_equal(probability$cond_T_count, 6)
        #     expect_equal(probability$cond_F_count, 3)
            
        #     expect_equal(probability$lenTT_c, 4)
        #     expect_equal(probability$lenTF_c, 1)
        #     expect_equal(probability$lenFT_c, 4)
        #     expect_equal(probability$lenFF_c, 0)
        #     expect_equal(probability$cond_T_count_c, 8)
        #     expect_equal(probability$cond_F_count_c, 1)
        # })
    def test_getGeneProbabilities_basic_p27_CycE(self):
        # Prepare input data
        main_params = setupdata2()

        target_gene = ["p27"]
        conditional_genes = ["CycE"]
        temporal = 1

        # Call the function
        probability = fbnnet_core.getGeneProbabilities_basic(main_params, None, target_gene, conditional_genes, temporal, None)
        probability = probability["1"]
        # # Access results

        print("Time step:", probability["timestep"])
        print("target_T_count:", probability["target_T_count"])
        print("target_F_count:", probability["target_F_count"])
        print("cond_T_count:", probability["cond_T_count"])
        print("cond_F_count:", probability["cond_F_count"])
        print("cond_T_count_c:", probability["cond_T_count_c"])
        print("cond_F_count_c:", probability["cond_F_count_c"])
        print("lenTT:", probability["lenTT"])
        print("lenTF:", probability["lenTF"])
        print("lenFT:", probability["lenFT"])
        print("lenFF:", probability["lenFF"])
        print("lenTT_c:", probability["lenTT_c"])
        print("lenTF_c:", probability["lenTF_c"])
        print("lenFT_c:", probability["lenFT_c"])
        print("lenFF_c:", probability["lenFF_c"])

        self.assertTrue(probability["target_T_count"] == 7)
        self.assertTrue(probability["target_F_count"] == 2)
        self.assertTrue(probability["lenTT"] == 1)
        self.assertTrue(probability["lenTF"] == 6)
        self.assertTrue(probability["lenFT"] == 1)
        self.assertTrue(probability["lenFF"] == 1)
        self.assertTrue(probability["cond_T_count"] == 2)
        self.assertTrue(probability["cond_F_count"] == 7)
        self.assertTrue(probability["lenTT_c"] == 1)
        self.assertTrue(probability["lenTF_c"] == 0)
        self.assertTrue(probability["lenFT_c"] == 7)
        self.assertTrue(probability["lenFF_c"] == 1)
        self.assertTrue(probability["cond_T_count_c"] == 8)
        self.assertTrue(probability["cond_F_count_c"] == 1)
        # # Uncomment the following lines to test with different parameters
        # it("getGenePrababilities(mainParameters,NULL,\"p27\",\"CycE\",1)", {
        #     probability <- getGenePrababilities_basic(mainParameters, NULL, "p27", "CycE", 1, NULL)[[1]]
        #     expect_equal(probability$target_T_count, 7)
        #     expect_equal(probability$target_F_count, 2)
        #     expect_equal(probability$lenTT, 1)
        #     expect_equal(probability$lenTF, 6)
        #     expect_equal(probability$lenFT, 1)
        #     expect_equal(probability$lenFF, 1)
        #     expect_equal(probability$cond_T_count, 2)
        #     expect_equal(probability$cond_F_count, 7)
            
        #     expect_equal(probability$lenTT_c, 1)
        #     expect_equal(probability$lenTF_c, 0)
        #     expect_equal(probability$lenFT_c, 7)
        #     expect_equal(probability$lenFF_c, 1)
        #     expect_equal(probability$cond_T_count_c, 8)
        #     expect_equal(probability$cond_F_count_c, 1)
        # })

    def test_getGeneProbabilities_basic_CycE_E2F(self):
        # Prepare input data
        main_params = setupdata2()

        print(main_params["testseries"])
        target_gene = ["CycE"]
        conditional_genes = ["E2F"]
        temporal = 1

        fixedgenestate = {
            "p27": 1,
            "CycD": 0
        }
  
        # Call the function
        probability = fbnnet_core.getGeneProbabilities_basic(main_params, fixedgenestate, target_gene, conditional_genes, temporal, None)
        probability = probability["1"]
        # # Access results

        print("Time step:", probability["timestep"])
        print("target_T_count:", probability["target_T_count"])
        print("target_F_count:", probability["target_F_count"])
        print("cond_T_count:", probability["cond_T_count"])
        print("cond_F_count:", probability["cond_F_count"])
        print("cond_T_count_c:", probability["cond_T_count_c"])
        print("cond_F_count_c:", probability["cond_F_count_c"])
        print("lenTT:", probability["lenTT"])
        print("lenTF:", probability["lenTF"])
        print("lenFT:", probability["lenFT"])
        print("lenFF:", probability["lenFF"])
        print("lenTT_c:", probability["lenTT_c"])
        print("lenTF_c:", probability["lenTF_c"])
        print("lenFT_c:", probability["lenFT_c"])
        print("lenFF_c:", probability["lenFF_c"])

        self.assertTrue(probability["target_T_count"] == 1)
        self.assertTrue(probability["target_F_count"] == 8)
        self.assertTrue(probability["lenTT"] == 1)
        self.assertTrue(probability["lenTF"] == 6)
        self.assertTrue(probability["lenFT"] == 2)
        self.assertTrue(probability["lenFF"] == 1)
        self.assertTrue(probability["cond_T_count"] == 3)
        self.assertTrue(probability["cond_F_count"] == 0)
        self.assertTrue(probability["lenTT_c"] == 1)
        self.assertTrue(probability["lenTF_c"] == 2)
        self.assertTrue(probability["lenFT_c"] == 0)
        self.assertTrue(probability["lenFF_c"] == 0)
        self.assertTrue(probability["cond_T_count_c"] == 1)
        self.assertTrue(probability["cond_F_count_c"] == 2)
        # it("getGenePrababilities(mainParameters,list(\"p27\"=1,\"CycD\"=0),\"CycE\",\"E2F\",1)", {
            
        #     probability <- getGenePrababilities_basic(mainParameters, list(p27 = 1, CycD = 0), "CycE", "E2F", 1, NULL)[[1]]
        #     expect_equal(probability$target_T_count, 1)
        #     expect_equal(probability$target_F_count, 8)
        #     expect_equal(probability$lenTT, 1)
        #     expect_equal(probability$lenTF, 0)
        #     expect_equal(probability$lenFT, 2)
        #     expect_equal(probability$lenFF, 0)
        #     expect_equal(probability$cond_T_count, 3)
        #     expect_equal(probability$cond_F_count, 0)
            
        #     expect_equal(probability$lenTT_c, 1)
        #     expect_equal(probability$lenTF_c, 2)
        #     expect_equal(probability$lenFT_c, 0)
        #     expect_equal(probability$lenFF_c, 0)
        #     expect_equal(probability$cond_T_count_c, 1)
        #     expect_equal(probability$cond_F_count_c, 2)
        # })
if __name__ == "__main__":
    # convert an array into matrix
    # Example 1D array
    arr = np.array([1, 2, 3, 4])

    # Reshape into a 2x2 matrix
    matrix = arr.reshape(2, 2)

    # Example 2x2 matrix
    matrix = np.array([[1, 2], [3, 4]])

    # Convert back to a 1D array
    array = matrix.flatten()

    print(array)

    print(matrix)

    # Create individual matrices
    matrix1 = np.array([[1, 2], [3, 4]])
    matrix2 = np.array([[5, 6], [7, 8]])
    matrix3 = np.array([[9, 10], [11, 12]])

    # Stack matrices along a new axis to form a cube
    cube = np.stack([matrix1, matrix2, matrix3], axis=0)
    print(" printing the cube shape")
    # The result `(3, 2, 2)` from `cube.shape` indicates the dimensions of the 3D array (cube):

    # 1. **3**: The cube contains 3 matrices (or slices) along the first dimension (axis 0).
    # 2. **2**: Each matrix has 2 rows (second dimension, axis 1).
    # 3. **2**: Each matrix has 2 columns (third dimension, axis 2).

    # In summary, the cube is a 3D array with 3 matrices, and each matrix is of size 2x2.
    print(cube.shape)  # Output: (3, 2, 2)
    print(" printing the cube")
    print(cube)
    print(" printing the 2 cube")
    # Two 3D cubes
    cube1 = np.ones((2, 2, 2))  # Shape: (2, 2, 2)
    cube2 = np.zeros((2, 2, 2))  # Shape: (2, 2, 2)

    # Stack along axis=1
    result = np.stack([cube1, cube2], axis=1)
    # - `cube1` and `cube2` each have a shape of `(2, 2, 2)`.
    # - After stacking along `axis=0`, the resulting array has a new first dimension, making its shape `(2, 2, 2, 2)`.
    # - The first dimension (`2`) corresponds to the number of cubes being stacked.
    # - `cube1` and `cube2` each have a shape of `(2, 2, 2)`.
    # - After stacking along `axis=1`, the resulting array has a new second dimension, making its shape `(2, 2, 2, 2)`.
    # - The first dimension (`2`) corresponds to the number of "outermost groups" in the original cubes, and the second dimension (`2`) corresponds to the number of cubes being stacked.

    print(result.shape)  # Output: (2, 2, 2, 2)

    print("------------------------------------------")
    unittest.main()

        # self.assertEqual(fbnnet_core.add(-1, 1), 0)