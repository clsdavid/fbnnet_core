#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <sstream>
#include <cmath>
#include "fbn_core.h"
#include "fbn_tree.h"
#include "fbn_utils.h"

namespace py = pybind11;

std::vector<std::string> filterTargetGenesByConditionGenes(
      std::vector<std::string>& targetGenes,
      py::dict& mainParameters,
      std::vector<std::string>& genes,
      py::object& prefix,
      int temporal)
{
   size_t tlen = targetGenes.size();
   size_t len = genes.size();
   std::vector<std::string> filteredTargetGene;
   
   for(size_t t=0; t<tlen; t++){
      std::vector<std::string> targetGene = {targetGenes[t]};

      for(size_t i=0; i<len; i++){
         std::vector<std::string> gene = {genes[i]};

         py::dict probabilityOfFourCombines = getGeneProbabilities(
             mainParameters, 
             prefix, 
             targetGene, 
             gene, 
             temporal, 
             false);
         
         if(probabilityOfFourCombines.is_none())
         {
            continue;
         }
         
         py::dict probabilityOfFourCombines_P;
         py::dict probabilityOfFourCombines_N;

         probabilityOfFourCombines_P = probabilityOfFourCombines["BestFitP"];
         probabilityOfFourCombines_N = probabilityOfFourCombines["BestFitN"];    

         if(!probabilityOfFourCombines_P["is_essential_gene"].cast<bool>() && 
            !probabilityOfFourCombines_N["is_essential_gene"].cast<bool>()){
            continue;
         }

         filteredTargetGene.push_back(targetGene[0]);
         break;
      }
   }
   return filteredTargetGene;
}

// Main function
py::dict getGeneProbabilities_measurements(
    std::vector<std::string>& targetGene,
    py::dict& mainParameters,
    std::vector<std::string>& genes,
    py::object& prefix,
    int temporal,
    bool show_basic_measures)
{
    size_t len = genes.size();
    py::dict result_dict;
    for(size_t i = 0; i < len; i++) {
        std::vector<std::string> gene = {genes[i]};
        // py::print("Processing gene:", genes[i]);

        py::dict probabilityOfFourCombines = getGeneProbabilities(
            mainParameters,
            prefix,
            targetGene,
            gene,
            temporal,
            show_basic_measures
        );

        // py::print("probabilityOfFourCombines:", probabilityOfFourCombines);
        if(probabilityOfFourCombines.is_none()) {
            continue;
        }

        py::dict probabilityOfFourCombines_P;
        py::dict probabilityOfFourCombines_N;

        probabilityOfFourCombines_P = probabilityOfFourCombines["BestFitP"];
        probabilityOfFourCombines_N = probabilityOfFourCombines["BestFitN"];

        // Check essential gene conditions
        if(!probabilityOfFourCombines_P["is_essential_gene"].cast<bool>() &&
           !probabilityOfFourCombines_N["is_essential_gene"].cast<bool>()) {
            continue;
        }

        if(!probabilityOfFourCombines_P["is_essential_gene"].cast<bool>()) {
            probabilityOfFourCombines_P = probabilityOfFourCombines_N;
        }

        if(!probabilityOfFourCombines_N["is_essential_gene"].cast<bool>()) {
            probabilityOfFourCombines_N = probabilityOfFourCombines_P;
        }

        // Check correlation conditions
        if(!probabilityOfFourCombines_P["isPositiveCorrelated"].cast<bool>() &&
           !probabilityOfFourCombines_P["isNegativeCorrelated"].cast<bool>()) {
            probabilityOfFourCombines_P = probabilityOfFourCombines_N;
        }

        if(!probabilityOfFourCombines_N["isPositiveCorrelated"].cast<bool>() &&
           !probabilityOfFourCombines_N["isNegativeCorrelated"].cast<bool>()) {
            
            bool all_equal = (probabilityOfFourCombines_P.equal(probabilityOfFourCombines_N));
            if(all_equal) {
                continue;
            }
            probabilityOfFourCombines_N = probabilityOfFourCombines_P;
        }

        // Prepare the result dictionary
        py::dict gene_result;
        gene_result["probabilityOfFourCombines_P"] = probabilityOfFourCombines_P;
        gene_result["probabilityOfFourCombines_N"] = probabilityOfFourCombines_N;

        // Add to main result dictionary with gene name as key
        result_dict[genes[i].c_str()] = gene_result;
    }

    // Set names attribute (equivalent to R's names(res) <- names)
    return result_dict;
}

py::dict buildProbabilityTreeOnTargetGene(
    std::vector<std::string>& targetGene,
    py::dict& mainParameters,
    std::vector<std::string>& genes,
    py::object& matchedgenes,
    py::object& matchedexpression,
    int maxK,
    int temporal,
    bool show_basic_measures,
    bool findPositiveRegulate,
    bool findNegativeRegulate) {
    
    //py::print("Building probability tree on target gene:", targetGene[0]);
    // Get measurements
    py::dict measurements = getGeneProbabilities_measurements(
        targetGene, mainParameters, genes, matchedgenes, temporal, show_basic_measures);
    
    std::vector<std::string> new_genes;
    for (auto item : measurements) {
        new_genes.push_back(item.first.cast<std::string>());
    }
    
    std::vector<std::string> unprocessedGenes = new_genes;
    std::vector<std::string> processedGenes;
    std::vector<py::dict> res(new_genes.size());
    
    if (maxK > static_cast<int>(unprocessedGenes.size())) {
        maxK = static_cast<int>(unprocessedGenes.size());
    }
    
    //py::print("Debug MaxK:", maxK);
    for (size_t i = 0; i < new_genes.size(); ++i) {
        // hard copy maxK to pmaxK
        // because maxK is changed in the loop
        int pmaxK = maxK;
        
        std::string gene = new_genes[i];
        std::vector<std::string> vgene = {gene};
        //py::print("Debug Processing gene:", gene);
        std::vector<size_t> unprocessed_index = a_not_in_b_index(unprocessedGenes, vgene);
        std::vector<std::string> temp_unprocessed;
        for (auto idx : unprocessed_index) {
            temp_unprocessed.push_back(unprocessedGenes[idx]);
        }
        unprocessedGenes = temp_unprocessed;
        
        py::dict newMatchedGenesT;
        py::dict newMatchedGenesF;
        std::vector<std::string> preprocessed;
        
        if (!matchedgenes.is_none()) {
            // IMPORTANT: `.cast<py::dict>()` on an object that already IS a
            // dict does not clone it - it returns another handle to the SAME
            // underlying Python dict. Assigning that same aliased dict to both
            // newMatchedGenesT and newMatchedGenesF (and to the shared
            // `matchedgenes` parameter) means later in-place mutations like
            // `newMatchedGenesT[gene] = 1` would corrupt newMatchedGenesF (and
            // vice versa), and would also leak into subsequent sibling
            // iterations of the enclosing loop since `matchedgenes` itself
            // would get mutated. Build independent copies instead.
            py::dict src = matchedgenes.cast<py::dict>();
            for (auto item : src) {
                newMatchedGenesT[item.first] = item.second;
                newMatchedGenesF[item.first] = item.second;
                preprocessed.push_back(item.first.cast<std::string>());
            }
        }
        
        std::vector<std::string> expressionT;
        std::vector<std::string> expressionF;
        //py::print("Debug Matched genes:", matchedgenes);
        if (!matchedexpression.is_none()) {
            std::vector<std::string> pmatchedexpression = matchedexpression.cast<std::vector<std::string>>();
            expressionT = {"&", gene};
            expressionF = {"&", "!", gene};
            expressionT = concatenate(pmatchedexpression, expressionT);
            expressionF = concatenate(pmatchedexpression, expressionF);
            
            if (!newMatchedGenesT.contains(gene.c_str())) {
                newMatchedGenesT[gene.c_str()] = 1;
            }
            
            if (!newMatchedGenesF.contains(gene.c_str())) {
                newMatchedGenesF[gene.c_str()] = 0;
            }
        } else {
            newMatchedGenesT[gene.c_str()] = 1;
            newMatchedGenesF[gene.c_str()] = 0;
            expressionT = {gene};
            expressionF = {"!", gene};
        }
        
        std::string expT = mpaste(expressionT);
        std::string expF = mpaste(expressionF);
        //py::print("Debug Expression T:", expT);
        //py::print("Debug Expression F:", expF);
        std::vector<std::string> inputgenes;
        for (auto item : newMatchedGenesT) {
            inputgenes.push_back(item.first.cast<std::string>());
        }
        inputgenes = char_sort(inputgenes, true);
        
        if (!preprocessed.empty() && std::find(preprocessed.begin(), preprocessed.end(), gene) != preprocessed.end()) {
            continue;
        }
        
        py::dict measuement = measurements[gene.c_str()].cast<py::dict>();
        py::dict probabilityOfFourCombines_P = measuement["probabilityOfFourCombines_P"].cast<py::dict>();
        py::dict probabilityOfFourCombines_N = measuement["probabilityOfFourCombines_N"].cast<py::dict>();
    
        
        double bestFit_P = probabilityOfFourCombines_P["bestFitP"].cast<double>();
        bool is_essential_gene_P = probabilityOfFourCombines_P["is_essential_gene"].cast<bool>();
        double bestFit_N = probabilityOfFourCombines_N["bestFitN"].cast<double>();
        bool is_essential_gene_N = probabilityOfFourCombines_N["is_essential_gene"].cast<bool>();
        std::string sign_P = probabilityOfFourCombines_P["signal_sign_T"].cast<std::string>();
        std::string sign_N = probabilityOfFourCombines_N["signal_sign_F"].cast<std::string>();
        
        // Find next branch
        py::dict subresultT;
        py::dict subresultF;
        
        std::vector<std::string> exlcudedSubgenes;
        std::vector<std::string> nextGenes_T;
        std::vector<std::string> nextGenes_F;
        
        //py::print("Debug unprocessed genes:", unprocessedGenes);
        //py::print("Debug input genes:", inputgenes);
        //py::print("Debug new matched genes T:", newMatchedGenesT);
        //py::print("Debug new matched genes F:", newMatchedGenesF);
        if (pmaxK > 1 && !(findPositiveRegulate && findNegativeRegulate)) {
            findPositiveRegulate = findPositiveRegulate || bestFit_P == 0;
            findNegativeRegulate = findNegativeRegulate || bestFit_N == 0;
            pmaxK = pmaxK - 1;
            
            exlcudedSubgenes = inputgenes;
            std::vector<size_t> unprocessed_index2 = a_not_in_b_index(unprocessedGenes, exlcudedSubgenes);
            for (auto idx : unprocessed_index2) {
                nextGenes_T.push_back(unprocessedGenes[idx]);
            }
            
            if (!nextGenes_T.empty() && is_essential_gene_P && (bestFit_P > 0 || bestFit_N > 0)) {
                py::object v_expT = py::cast(std::vector<std::string>{expT});
                //print the arguments
                // py::print("Debug buildProbabilityTreeOnTargetGene T arguments:");
                // py::print("targetGene:", targetGene);
                // py::print("nextGenes_T:", nextGenes_T);
                // py::print("newMatchedGenesT:", newMatchedGenesT);
                // py::print("v_expT:", v_expT);
                // py::print("pmaxK:", pmaxK);
                // py::print("temporal:", temporal);
                // py::print("show_basic_measures:", show_basic_measures);
                // py::print("findPositiveRegulate:", findPositiveRegulate);
                // py::print("findNegativeRegulate:", findNegativeRegulate);
                // Call the function
                subresultT = buildProbabilityTreeOnTargetGene(
                    targetGene, mainParameters, nextGenes_T, newMatchedGenesT, v_expT, 
                    pmaxK, temporal, show_basic_measures, findPositiveRegulate, findNegativeRegulate);
            }
            
            exlcudedSubgenes = inputgenes;
            unprocessed_index2 = a_not_in_b_index(unprocessedGenes, exlcudedSubgenes);
            for (auto idx : unprocessed_index2) {
                nextGenes_F.push_back(unprocessedGenes[idx]);
            }
            
            if (!nextGenes_F.empty() && is_essential_gene_N && (bestFit_P > 0 || bestFit_N > 0)) {
                py::object v_expF = py::cast(std::vector<std::string>{expF});
                //print the arguments
                // py::print("Debug buildProbabilityTreeOnTargetGene F arguments:");
                // py::print("targetGene:", targetGene);
                // py::print("nextGenes_F:", nextGenes_F);
                // py::print("newMatchedGenesF:", newMatchedGenesF);
                // py::print("v_expF:", v_expF);
                // py::print("pmaxK:", pmaxK);
                // py::print("temporal:", temporal);
                // py::print("show_basic_measures:", show_basic_measures);
                // py::print("findPositiveRegulate:", findPositiveRegulate);
                // py::print("findNegativeRegulate:", findNegativeRegulate);
                // Call the function
                subresultF = buildProbabilityTreeOnTargetGene(
                    targetGene, mainParameters, nextGenes_F, newMatchedGenesF, v_expF, 
                    pmaxK, temporal, show_basic_measures, findPositiveRegulate, findNegativeRegulate);
            }
        }
        
        // py::print("Debug subresultT:", subresultT);
        // py::print("Debug subresultF:", subresultF);
        // Positive regulation
        double pickT_support = probabilityOfFourCombines_P["pickT_support"].cast<double>();
        double pickT_causality_test = probabilityOfFourCombines_P["pickT_causality_test"].cast<double>();
        double pickT_value = probabilityOfFourCombines_P["Signal_P"].cast<double>();
        double pickT_noise = probabilityOfFourCombines_P["Noise_P"].cast<double>();
        double pickT_confidenceCounter = probabilityOfFourCombines_P["pickT_confidenceCounter"].cast<double>();
        double pickT_all_confidence = probabilityOfFourCombines_P["pickT_all_confidence"].cast<double>();
        double pickT_max_confidence = probabilityOfFourCombines_P["pickT_max_confidence"].cast<double>();
        bool isNegativeCorrelated_T = probabilityOfFourCombines_P["isNegativeCorrelated"].cast<bool>();
        bool isPositiveCorrelated_T = probabilityOfFourCombines_P["isPositiveCorrelated"].cast<bool>();
        int timestep_T = probabilityOfFourCombines_P["timestep"].cast<int>();
        double bestFitP = probabilityOfFourCombines_P["bestFitP"].cast<double>();
        double p_value_P = probabilityOfFourCombines_P["p_value"].cast<double>();
        double pickT_mutualInfo = probabilityOfFourCombines_P["pickT_mutualInfo"].cast<double>();
        
        // Negative regulation
        double pickF_support = probabilityOfFourCombines_N["pickF_support"].cast<double>();
        double pickF_causality_test = probabilityOfFourCombines_N["pickF_causality_test"].cast<double>();
        double pickF_value = probabilityOfFourCombines_N["Signal_N"].cast<double>();
        double pickF_noise = probabilityOfFourCombines_N["Noise_N"].cast<double>();
        double pickF_ConfidenceCounter = probabilityOfFourCombines_N["pickF_confidenceCounter"].cast<double>();
        double pickF_all_confidence = probabilityOfFourCombines_N["pickF_all_confidence"].cast<double>();
        double pickF_max_confidence = probabilityOfFourCombines_N["pickF_max_confidence"].cast<double>();
        bool isNegativeCorrelated_F = probabilityOfFourCombines_N["isNegativeCorrelated"].cast<bool>();
        bool isPositiveCorrelated_F = probabilityOfFourCombines_N["isPositiveCorrelated"].cast<bool>();
        int timestep_F = probabilityOfFourCombines_N["timestep"].cast<int>();
        double bestFitN = probabilityOfFourCombines_N["bestFitN"].cast<double>();
        double p_value_N = probabilityOfFourCombines_N["p_value"].cast<double>();
        double pickF_mutualInfo = probabilityOfFourCombines_N["pickF_mutualInfo"].cast<double>();
        
        std::string pick_expT;
        std::string pick_expF;
        std::string identity_T;
        std::string identity_F;
        std::vector<std::string> pattern;
        
        // py::print("Debug sign_P:", sign_P);
        // py::print("Debug sign_N:", sign_N);
        // py::print("Debug pick_expT:", pick_expT);
        // py::print("Debug pick_expF:", pick_expF);
        if (sign_P == "TT") {
            for (const auto& inputgene : inputgenes) {
                int gene_state = newMatchedGenesT[inputgene.c_str()].cast<int>();
                std::string this_gene = inputgene + "$" + std::to_string(gene_state);
                pattern.push_back(this_gene);
            }
            identity_T = mpaste(pattern, "_");
            pick_expT = expT;
        } else if (sign_P == "TF") {
            for (const auto& inputgene : inputgenes) {
                int gene_state = newMatchedGenesF[inputgene.c_str()].cast<int>();
                std::string this_gene = inputgene + "$" + std::to_string(gene_state);
                pattern.push_back(this_gene);
            }
            identity_T = mpaste(pattern, "_");
            pick_expT = expF;
        }
        // py::print("debug load pattern for activator");
        pattern.clear();
        pattern.push_back(identity_T);
        pattern.push_back("Activator_of");
        pattern.push_back(targetGene[0]);
        identity_T = mpaste(pattern, "_");
        // py::print("Debug activator");
        py::dict activator;
        activator["factor"] = pick_expT;
        activator["Confidence"] = std::to_string(pickT_value);
        activator["ConfidenceCounter"] = std::to_string(pickT_confidenceCounter);
        activator["all_confidence"] = std::to_string(pickT_all_confidence);
        activator["max_confidence"] = std::to_string(pickT_max_confidence);
        activator["support"] = std::to_string(pickT_support);
        activator["causality_test"] = std::to_string(pickT_causality_test);
        activator["Noise"] = std::to_string(pickT_noise);
        activator["Identity"] = identity_T;
        activator["type"] = sign_P;
        activator["timestep"] = std::to_string(timestep_T);
        activator["isNegativeCorrelated"] = std::to_string(isNegativeCorrelated_T);
        activator["isPositiveCorrelated"] = std::to_string(isPositiveCorrelated_T);
        activator["bestFitP"] = std::to_string(bestFitP);
        activator["p_value"] = std::to_string(p_value_P);
        activator["mutualInfo"] = std::to_string(pickT_mutualInfo);
        
        pattern.clear();
        if (sign_N == "FT") {
            for (const auto& inputgene : inputgenes) {
                int gene_state = newMatchedGenesT[inputgene.c_str()].cast<int>();
                std::string this_gene = inputgene + "$" + std::to_string(gene_state);
                pattern.push_back(this_gene);
            }
            identity_F = mpaste(pattern, "_");
            pick_expF = expT;
        } else if (sign_N == "FF") {
            for (const auto& inputgene : inputgenes) {
                int gene_state = newMatchedGenesF[inputgene.c_str()].cast<int>();
                std::string this_gene = inputgene + "$" + std::to_string(gene_state);
                pattern.push_back(this_gene);
            }
            identity_F = mpaste(pattern, "_");
            pick_expF = expF;
        }

        // py::print("debug load pattern for inhibitor");
        pattern.clear();
        pattern.push_back(identity_F);
        pattern.push_back("Inhibitor_of");
        pattern.push_back(targetGene[0]);

        identity_F = mpaste(pattern, "_");
        // py::print("Debug Inhibitor");
        py::dict inhibitor;
        inhibitor["factor"] = pick_expF;
        inhibitor["Confidence"] = std::to_string(pickF_value);
        inhibitor["ConfidenceCounter"] = std::to_string(pickF_ConfidenceCounter);
        inhibitor["all_confidence"] = std::to_string(pickF_all_confidence);
        inhibitor["max_confidence"] = std::to_string(pickF_max_confidence);
        inhibitor["support"] = std::to_string(pickF_support);
        inhibitor["causality_test"] = std::to_string(pickF_causality_test);
        inhibitor["Noise"] = std::to_string(pickF_noise);
        inhibitor["Identity"] = identity_F;
        inhibitor["type"] = sign_N;
        inhibitor["timestep"] = std::to_string(timestep_F);
        inhibitor["isNegativeCorrelated"] = std::to_string(isNegativeCorrelated_F);
        inhibitor["isPositiveCorrelated"] = std::to_string(isPositiveCorrelated_F);
        inhibitor["bestFitN"] = std::to_string(bestFitN);
        inhibitor["p_value"] = std::to_string(p_value_N);
        inhibitor["mutualInfo"] = std::to_string(pickF_mutualInfo);
        
        py::dict a_i_tor;
        a_i_tor["Activator"] = activator;
        a_i_tor["Inhibitor"] = inhibitor;
        
        py::dict in_res;
        in_res["ActivatorAndInhibitor"] = a_i_tor;
        in_res["Input"] = inputgenes;
        
        if (!subresultT.empty()) {
            in_res["SubGenesT"] = subresultT;
        }
        
        if (!subresultF.empty()) {
            in_res["SubGenesF"] = subresultF;
        }
        // py::print("Debug in_res:", in_res);
        res[i] = in_res;
    }
    
    // py::print("Debug res:", res);
    // Remove empty elements
    py::dict result_dict;
    for (size_t i = 0; i < res.size(); ++i) {
        if (!res[i].is_none()) {
            result_dict[new_genes[i].c_str()] = res[i];
        }
    }
    
    return result_dict;
}


py::dict process_cube_algorithm(
    std::string& target_gene,
    std::vector<std::string>& conditional_genes,
    int maxK,
    int temporal,
    py::dict& main_parameters,
    py::object& matchedgenes,
    py::object& matchedexpression
) {
    // Create the nested result structure
    py::dict res;
    py::dict sub_res;
    std::vector<std::string> target_genes;
    target_genes.push_back(target_gene);
    // Call the core function
    sub_res["SubGenes"] = buildProbabilityTreeOnTargetGene(
        target_genes,
        main_parameters,
        conditional_genes,
        matchedgenes,
        matchedexpression,
        maxK,
        temporal,
        false,
        false,
        false
    );
    
    // Set the target gene as the key in the result dictionary
    res[target_gene.c_str()] = sub_res;
    
    return res;
}

// PyBind11 module definition
PYBIND11_MODULE(_tree, m) {
    m.def("filterTargetGenesByConditionGenes", &filterTargetGenesByConditionGenes,
        "Filter target genes based on condition genes",
        py::arg("targetGenes"),
        py::arg("mainParameters"),
        py::arg("genes"),
        py::arg("prefix") = py::none(),
        py::arg("temporal") = 1);
    m.def("getGeneProbabilities_measurements", &getGeneProbabilities_measurements,
        "Get the main measurements based on the input data",
        py::arg("targetGene"),
        py::arg("mainParameters"),
        py::arg("genes"),
        py::arg("prefix") = py::none(),
        py::arg("temporal") = 1,
        py::arg("show_basic_measures") = false);

    m.def("buildProbabilityTreeOnTargetGene", &buildProbabilityTreeOnTargetGene,
        "Build a probability tree on the target gene",
        py::arg("targetGene"),
        py::arg("mainParameters"),
        py::arg("genes"),
        py::arg("matchedgenes") = py::none(),
        py::arg("matchedexpression") = py::none(),
        py::arg("maxK") = 4,
        py::arg("temporal") = 1,
        py::arg("show_basic_measures") = false,
        py::arg("findPositiveRegulate") = false,
        py::arg("findNegativeRegulate") = false);

    m.def("process_cube_algorithm", &process_cube_algorithm,
        "Process the cube algorithm for gene regulation",
        py::arg("target_gene"),
        py::arg("conditional_genes"),
        py::arg("maxK"),
        py::arg("temporal"),
        py::arg("main_parameters"),
        py::arg("matchedgenes") = py::none(),
        py::arg("matchedexpression") = py::none());
    m.attr("__version__") = "0.1.0";
}