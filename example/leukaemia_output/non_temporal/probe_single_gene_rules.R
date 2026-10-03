# Run in R:  Rscript probe_single_gene_rules.R > r_probe.txt
# For single-gene rules (target <- one conditional gene, maxK = 1, temporal = 1) that Python keeps but
# R's final network lacks (MISSING) and for similar ones R keeps (KEPT): is the gene in the cube root
# (SubGenes) and which measures does R compute? Send r_probe.txt back.
suppressMessages(library(FBNNet))
timeseries <- NULL
data("Leukeamia_Timeseries")
timeseries <- Leukeamia_Timeseries

pairs <- list(
  c("ABHD17B", "HRK"),     # MISSING in R, table (16,0,18,8),  Fisher p 0.01587
  c("ABHD17B", "EMP1"),    # MISSING in R, table (15,0,19,8),  p 0.0352
  c("IL27RA",  "FGL2"),    # MISSING in R, table (8,0,18,16),  p 0.01587
  c("S100A8",  "LGALS3"),  # MISSING in R, table (8,0,18,16),  p 0.01587
  c("CCR1",    "EPPK1"),   # MISSING in R, table (4,0,13,25),  p 0.02126
  c("ABHD17B", "B3GNT2"),  # KEPT, table (23,0,11,8), p 0.00064
  c("SESN1",   "PPBP"),    # KEPT, table (9,0,18,15), p 0.01598
  c("WASF1",   "P2RX5"),   # KEPT, table (15,0,18,9), p 0.01598
  c("STAB1",   "EPPK1"),   # KEPT, table (4,0,12,26), p 0.01626
  c("TRIB1",   "IL18RAP")  # KEPT, table (8,0,17,17), p 0.01349
)

for (p in pairs) {
  target <- p[1]; cond <- p[2]
  cube <- constructFBNCube(target_genes = target, conditional_genes = cond,
                           timeseriesCube = timeseries, maxK = 1, temporal = 1, useParallel = FALSE)
  cat("\n==========", target, "<-", cond, "==========\n")
  sub <- cube[[target]]$SubGenes
  cat("in root pool:", cond %in% names(sub), "\n")
  if (cond %in% names(sub)) {
    m <- sub[[cond]]$ActivatorAndInhibitor
    cat("-- Activator --\n");  print(m$Activator)
    cat("-- Inhibitor --\n");  print(m$Inhibitor)
  }
}
