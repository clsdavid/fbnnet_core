#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include "fbn_utils.h"

namespace py = pybind11;

// Helper type for nullable parameters
using NullableList = py::object;
using NullableStringVec = py::object;

// -------------------- filterTargetGenesByConditionGenes --------------------
std::vector<std::string> filterTargetGenesByConditionGenes(
    const std::vector<std::string>& targetGenes,
    py::dict mainParameters,
    const std::vector<std::string>& genes,
    const NullableList& matchedgenes,
    int temporal = 1,
    const NullableList& targetCounts = py::none()) 
{
    std::vector<std::string> filteredTargetGene;
    
    for(const auto& targetGene : targetGenes) {
        for(const auto& gene : genes) {
            auto probability = getGenePrababilities(
                mainParameters,
                matchedgenes,
                std::vector<std::string>{targetGene},
                std::vector<std::string>{gene},
                temporal,
                targetCounts
            );
            
            if(probability.is_none()) continue;

            auto probP = probability["getBestFitP"].cast<py::dict>();
            auto probN = probability["getBestFitN"].cast<py::dict>();
            
            if(!probP["is_essential_gene"].cast<bool>() && 
               !probN["is_essential_gene"].cast<bool>()) continue;

            filteredTargetGene.push_back(targetGene);
            break;
        }
    }
    return filteredTargetGene;
}

// -------------------- getGenePrababilities_measurements --------------------
py::dict getGenePrababilities_measurements(
    const std::string& targetGene,
    py::dict mainParameters,
    const std::vector<std::string>& genes,
    const NullableList& matchedgenes,
    int temporal = 1,
    const NullableList& targetCounts = py::none()) 
{
    py::dict res;
    
    for(const auto& gene : genes) {
        auto probability = getGenePrababilities(
            mainParameters,
            matchedgenes,
            std::vector<std::string>{targetGene},
            std::vector<std::string>{gene},
            temporal,
            targetCounts
        );
        
        if(probability.is_none()) continue;

        py::dict probP = probability["getBestFitP"];
        py::dict probN = probability["getBestFitN"];
        
        if(!probP["is_essential_gene"].cast<bool>() && 
           !probN["is_essential_gene"].cast<bool>()) continue;

        // ... rest of processing logic ...
        
        py::dict entry;
        entry["probabilityOfFourCombines_P"] = probP;
        entry["probabilityOfFourCombines_N"] = probN;
        entry["targetCounts"] = probability["targetCounts"];
        
        res[gene.c_str()] = entry;
    }
    
    return remove_empty_elements(res);
}


py::dict buildProbabilityTreeOnTargetGene(
   const std::string& targetGene,
   py::dict mainParameters,
   const std::vector<std::string>& genes,
   const py::object& matchedgenes,
   const py::object& matchedexpression,
   int maxK = 4,
   int temporal = 1,
   const py::object& targetCounts = py::none(),
   bool findPositiveRegulate = false,
   bool findNegativeRegulate = false)
{
   // Get probability measurements
   auto measurements = getGenePrababilities_measurements(
       targetGene, mainParameters, genes, matchedgenes, temporal, targetCounts
   );
   
   py::list new_genes = measurements.attr("keys")();
   py::list unprocessedGenes = py::module::import("copy").attr("deepcopy")(new_genes);
   py::dict res;
   
   // Handle maxK constraint
   maxK = std::min(maxK, py::len(unprocessedGenes));

   for(size_t i=0; i<py::len(new_genes); i++) {
       std::string gene = new_genes[i].cast<std::string>();
       int current_maxK = maxK;

       // Remove processed gene
       unprocessedGenes.attr("remove")(gene);
       
       // Initialize matched genes
       py::dict newMatchedGenesT, newMatchedGenesF;
       std::vector<std::string> preprocessed;
       
       if(!matchedgenes.is_none()) {
           auto mg = matchedgenes.cast<py::dict>();
           newMatchedGenesT = mg;
           newMatchedGenesF = mg;
           preprocessed = get_dict_keys(mg);
       }

       // Build expressions
       std::vector<std::string> expressionT, expressionF;
       if(!matchedexpression.is_none()) {
           auto me = matchedexpression.cast<std::vector<std::string>>();
           expressionT = concatenator(me, {"&", gene});
           expressionF = concatenator(me, {"&", "!", gene});
           
           // Update matched genes
           if(!newMatchedGenesT.contains(gene)) {
               newMatchedGenesT[gene] = 1;
           }
           if(!newMatchedGenesF.contains(gene)) {
               newMatchedGenesF[gene] = 0;
           }
       } else {
           newMatchedGenesT[gene] = 1;
           newMatchedGenesF[gene] = 0;
           expressionT = {gene};
           expressionF = {"!", gene};
       }

       // Process measurements
       auto measurement = measurements[gene].cast<py::dict>();
       auto probP = measurement["probabilityOfFourCombines_P"].cast<py::dict>();
       auto probN = measurement["probabilityOfFourCombines_N"].cast<py::dict>();
       
       // Recursive processing
       py::dict subresultT, subresultF;
       if(current_maxK > 1 && !(findPositiveRegulate && findNegativeRegulate)) {
           // Get next genes for recursion
           auto nextGenesT = list_diff(unprocessedGenes, get_dict_keys(newMatchedGenesT));
           auto nextGenesF = list_diff(unprocessedGenes, get_dict_keys(newMatchedGenesF));
           
           if(!nextGenesT.empty()) {
               subresultT = buildProbabilityTreeOnTargetGene(
                   targetGene, mainParameters, 
                   nextGenesT.cast<std::vector<std::string>>(),
                   newMatchedGenesT,
                   py::cast(expressionT),
                   current_maxK-1,
                   temporal,
                   targetCounts,
                   findPositiveRegulate || probP["bestFitP"].cast<double>() == 0,
                   findNegativeRegulate
               );
           }
           
           if(!nextGenesF.empty()) {
               subresultF = buildProbabilityTreeOnTargetGene(
                   targetGene, mainParameters,
                   nextGenesF.cast<std::vector<std::string>>(),
                   newMatchedGenesF,
                   py::cast(expressionF),
                   current_maxK-1,
                   temporal,
                   targetCounts,
                   findPositiveRegulate,
                   findNegativeRegulate || probN["bestFitN"].cast<double>() == 0
               );
           }
       }

       // Build result entry
       py::dict entry;
       entry["activator"] = create_activator_entry(probP, newMatchedGenesT, expressionT, targetGene);
       entry["inhibitor"] = create_inhibitor_entry(probN, newMatchedGenesF, expressionF, targetGene);
       
       // Add subresults
       if(!subresultT.empty()) entry["subgenesT"] = subresultT;
       if(!subresultF.empty()) entry["subgenesF"] = subresultF;
       
       res[gene.c_str()] = entry;
   }
   
   return remove_empty_elements(res);
}

py::dict mineNetworksDirect(
   const std::string& targetGene,
   py::dict mainParameters,
   const std::vector<std::string>& genes,
   const py::object& matchedgenes,
   const py::object& matchedexpression,
   int maxK = 4,
   int temporal = 1,
   const py::object& targetCounts = py::none(),
   bool findPositiveRegulate = false,
   bool findNegativeRegulate = false)
{
   // Get probability measurements
   auto measurements = getGenePrababilities_measurements(
       targetGene, mainParameters, genes, matchedgenes, temporal, targetCounts
   );
   
   py::list new_genes = measurements.attr("keys")();
   py::list unprocessedGenes = py::module::import("copy").attr("deepcopy")(new_genes);
   
   py::dict result;
   py::list activators;
   py::list inhibitors;
   
   maxK = std::min(maxK, py::len(unprocessedGenes));

   for(size_t i=0; i<py::len(new_genes); i++) {
       std::string gene = new_genes[i].cast<std::string>();
       unprocessedGenes.attr("remove")(gene);
       
       // Initialize matched genes and expressions
       py::dict newMatchedGenesT, newMatchedGenesF;
       std::vector<std::string> preprocessed;
       
       if(!matchedgenes.is_none()) {
           auto mg = matchedgenes.cast<py::dict>();
           newMatchedGenesT = mg;
           newMatchedGenesF = mg;
           preprocessed = get_dict_keys(mg);
       }

       // Build expressions
       std::vector<std::string> expressionT, expressionF;
       if(!matchedexpression.is_none()) {
           auto me = matchedexpression.cast<std::vector<std::string>>();
           expressionT = concatenator(me, {"&", gene});
           expressionF = concatenator(me, {"&", "!", gene});
           
           if(!newMatchedGenesT.contains(gene)) {
               newMatchedGenesT[gene] = 1;
           }
           if(!newMatchedGenesF.contains(gene)) {
               newMatchedGenesF[gene] = 0;
           }
       } else {
           newMatchedGenesT[gene] = 1;
           newMatchedGenesF[gene] = 0;
           expressionT = {gene};
           expressionF = {"!", gene};
       }

       // Process measurements
       auto measurement = measurements[gene].cast<py::dict>();
       auto probP = measurement["probabilityOfFourCombines_P"].cast<py::dict>();
       auto probN = measurement["probabilityOfFourCombines_N"].cast<py::dict>();

       // Recursive processing
       py::dict subresultT, subresultF;
       int current_maxK = maxK;
       if(current_maxK > 1 && !(findPositiveRegulate && findNegativeRegulate)) {
           auto nextGenesT = list_diff(unprocessedGenes, get_dict_keys(newMatchedGenesT));
           auto nextGenesF = list_diff(unprocessedGenes, get_dict_keys(newMatchedGenesF));
           
           if(!nextGenesT.empty()) {
               subresultT = mineNetworksDirect(
                   targetGene, mainParameters,
                   nextGenesT.cast<std::vector<std::string>>(),
                   newMatchedGenesT,
                   py::cast(expressionT),
                   current_maxK-1,
                   temporal,
                   targetCounts,
                   findPositiveRegulate || probP["bestFitP"].cast<double>() == 0,
                   findNegativeRegulate
               );
           }
           
           if(!nextGenesF.empty()) {
               subresultF = mineNetworksDirect(
                   targetGene, mainParameters,
                   nextGenesF.cast<std::vector<std::string>>(),
                   newMatchedGenesF,
                   py::cast(expressionF),
                   current_maxK-1,
                   temporal,
                   targetCounts,
                   findPositiveRegulate,
                   findNegativeRegulate || probN["bestFitN"].cast<double>() == 0
               );
           }
       }

       // Create activator/inhibitor entries
       auto activator = create_activator_entry(probP, newMatchedGenesT, expressionT, targetGene);
       auto inhibitor = create_inhibitor_entry(probN, newMatchedGenesF, expressionF, targetGene);

       // Collect results
       if(probP["bestFitP"].cast<double>() == 0) {
           activators.append(activator);
       }
       if(probN["bestFitN"].cast<double>() == 0) {
           inhibitors.append(inhibitor);
       }

       // Merge subresults
       if(!subresultT.empty()) {
           activators = concatenate_lists(activators, subresultT["activators"]);
       }
       if(!subresultF.empty()) {
           inhibitors = concatenate_lists(inhibitors, subresultF["inhibitors"]);
       }
   }

   result["activators"] = remove_empty_elements(activators);
   result["inhibitors"] = remove_empty_elements(inhibitors);
   return result;
}

// [[Rcpp::export]]
Rcpp::List internalloopByWhole2(Rcpp::CharacterVector target_gene, 
                                Rcpp::CharacterVector conditional_genes, 
                                Rcpp::IntegerVector maxK, 
                                Rcpp::IntegerVector temporal, 
                                Rcpp::Environment mainParameters) {
   List res(1);
   List sub_res(1);
   sub_res[0] = buildProbabilityTreeOnTargetGene(target_gene, mainParameters, conditional_genes, R_NilValue, R_NilValue, maxK, temporal, R_NilValue);
   sub_res.attr("names") = CharacterVector::create("SubGenes");
   res[0] = sub_res;
   res.attr("names") = target_gene;
   return(res);
}

y::dict internalloopByWhole2(
   const std::string& target_gene,
   const std::vector<std::string>& conditional_genes,
   int maxK,
   int temporal,
   py::dict mainParameters) 
{
   py::dict res;
   py::dict sub_res;
   
   // Call the converted buildProbabilityTreeOnTargetGene
   py::dict probability_tree = buildProbabilityTreeOnTargetGene(
       target_gene,
       mainParameters,
       conditional_genes,
       py::none(),  // matchedgenes
       py::none(),  // matchedexpression
       maxK,
       temporal,
       py::none(),  // targetCounts
       false,       // findPositiveRegulate
       false        // findNegativeRegulate
   );
   
   sub_res["SubGenes"] = probability_tree;
   res[target_gene.c_str()] = sub_res;
   
   return res;
}

PYBIND11_MODULE(fbn_core, m) {
   m.def("filterTargetGenesByConditionGenes", &filterTargetGenesByConditionGenes,
      py::arg("targetGenes"), py::arg("mainParameters"),
      py::arg("genes"), py::arg("matchedgenes") = py::none(),
      py::arg("temporal") = 1, py::arg("targetCounts") = py::none());

   m.def("getGenePrababilities_measurements", &getGenePrababilities_measurements,
         py::arg("targetGene"), py::arg("mainParameters"),
         py::arg("genes"), py::arg("matchedgenes") = py::none(),
         py::arg("temporal") = 1, py::arg("targetCounts") = py::none());

   m.def("buildProbabilityTreeOnTargetGene", &buildProbabilityTreeOnTargetGene,
         py::arg("targetGene"), py::arg("mainParameters"),
         py::arg("genes"), py::arg("matchedgenes") = py::none(),
         py::arg("matchedexpression") = py::none(), py::arg("maxK") = 4,
         py::arg("temporal") = 1, py::arg("targetCounts") = py::none(),
         py::arg("findPositiveRegulate") = false,
         py::arg("findNegativeRegulate") = false);

   m.def("mineNetworksDirect", &mineNetworksDirect,
         py::arg("targetGene"), py::arg("mainParameters"),
         py::arg("genes"), py::arg("matchedgenes") = py::none(),
         py::arg("matchedexpression") = py::none(), py::arg("maxK") = 4,
         py::arg("temporal") = 1, py::arg("targetCounts") = py::none(),
         py::arg("findPositiveRegulate") = false,
         py::arg("findNegativeRegulate") = false);
   m.def("buildProbabilityTreeOnTargetGene", &buildProbabilityTreeOnTargetGene,
         py::arg("targetGene"),
         py::arg("mainParameters"),
         py::arg("genes"),
         py::arg("matchedgenes") = py::none(),
         py::arg("matchedexpression") = py::none(),
         py::arg("maxK") = 4,
         py::arg("temporal") = 1,
         py::arg("targetCounts") = py::none(),
         py::arg("findPositiveRegulate") = false,
         py::arg("findNegativeRegulate") = false);

   m.def("mineNetworksDirect", &mineNetworksDirect,
      py::arg("targetGene"),
      py::arg("mainParameters"),
      py::arg("genes"),
      py::arg("matchedgenes") = py::none(),
      py::arg("matchedexpression") = py::none(),
      py::arg("maxK") = 4,
      py::arg("temporal") = 1,
      py::arg("targetCounts") = py::none(),
      py::arg("findPositiveRegulate") = false,
      py::arg("findNegativeRegulate") = false);

   m.def("internalloopByWhole2", &internalloopByWhole2,
      py::arg("target_gene"),
      py::arg("conditional_genes"),
      py::arg("maxK"),
      py::arg("temporal"),
      py::arg("mainParameters"));

}