# Example: reproducing the R vignette experiment

This folder reproduces, end to end with this Python port, the worked example from the
original R package's vignette (`FBNNet2_public/vignettes/FBNNet.Rmd`, section
*"Extract the Fundamental Boolean Network"*) and its `tests/testthat/test-fbngraphic.R` /
`test-attractor.R` counterparts.

## Data

* **`example.bn`** — a byte-for-byte copy of the R package's `ExampleNetwork` test fixture
  (`FBNNet2_public/tests/testthat/example.bn`), a 5-gene BoolNet-style network:

  ```
  targets, factors
  Gene1, Gene1
  Gene2, Gene1 & Gene5 & !Gene4
  Gene3, Gene3
  Gene4, Gene3 & !(Gene1 & Gene5)
  Gene5, !Gene2
  ```

  This is also the same file used at the repository root (`../example.bn`); it's duplicated
  here so this folder is self-contained.

The R package's other bundled datasets (`BoolNet_CellCycle_Network`, `ExampleTimeseriesData`,
`FBNExampleNetworks`, `FBNcellcycleNetwork`, the `Leukeamia_*`/`Common_Genes_*` study data, ...)
are shipped as R `.rda` binary objects. Several of them (nested BoolNet S4 objects, lists of
matrices) can't be decoded without R itself — `pyreadr`/`librdata` fail on them
(`LibrdataError('Invalid file, or file has unsupported features')`) or return no top-level
data frame — and R is not available in this environment, so they weren't converted. `example.bn`
is the one dataset that both projects have in an identical, plain-text, directly comparable
form, and it's the dataset the vignette's fully worked example is actually built on.

## What the script does

`reproduce_vignette_experiment.py` mirrors the vignette's R code

```r
initialStates  <- generateAllCombinationBinary(ExampleNetwork$genes)
trainingseries <- genereateBoolNetTimeseries(ExampleNetwork, initialStates, 43, type = "synchronous")
FBNcellcyclenetwork <- generateFBMNetwork(timeseries_data = trainingseries, maxK = 4,
                                          max_deep_temporal = 1, useParallel = FALSE)
FBNNetwork.Graph(FBNcellcyclenetwork)
attractor <- searchForAttractors(FBNcellcyclenetwork, initialStates, genes)
FBNNetwork.Graph.DrawAttractor(FBNcellcyclenetwork, attractor, 2)
```

using only functions already ported into `py_src/`:

1. `py_src.boolnet.load_network` — load `example.bn`.
2. `py_src.data_utils.generateAllCombinationBinary` / `generateBoolNetTimeseries` — simulate
   43 synchronous time steps from every one of the $2^5=32$ possible initial states.
3. `py_src.application.generate_fbm_network` — mine a Fundamental Boolean Network from that
   data in one call (this is the Python port of `generateFBMNetwork`).
4. `py_src.network_graph.draw_static_network` / `plot_network` — render the mined network.
5. `py_src.attractor.search_for_attractors` — find its FBM attractors.
6. `py_src.network_graph.draw_attractor` — render one attractor's state cycle.

> **Note:** the vignette also calls `reconstructTimeseries`/`generateSimilaryReport` to round-trip
> simulate the mined network and report an accuracy score. That function does not actually exist
> anywhere in the current `FBNNet2_public/R/*.R` sources (only in the stale vignette/tests), so
> there was nothing to port here either — it's left out rather than guessed at.

## Running it

```bash
source .venv/bin/activate   # from the repository root
python example/reproduce_vignette_experiment.py
```

This (re)generates the files below. `get_fbm_successor` (used by attractor search) breaks ties
between competing rules at random — mirroring the "uncertainty" mechanism described in the FBM
paper — so the script seeds `random` before searching for attractors to keep the results
reproducible run to run.

| File | Contents |
|---|---|
| `mined_network_summary.txt` | Every activator/inhibitor rule mined for each gene, with its expression, timestep and confidence |
| `mined_network_graph.png` | The full mined network as a gene → gene graph (green = activator, dark red = inhibitor) |
| `mined_network_graph_gene1_forward.png` | A zoomed-in `gene(node) -> activator/inhibitor rule (timestep) -> gene(node)` view of what regulates `Gene1`, one level forward |
| `attractor_0.png` | The first FBM attractor found, as a genes × cycle-step on/off heatmap |

### Actual results (from the committed run)

Mining `example.bn`'s simulated time series with `max_k=4`, `max_deep_temporal=1` recovers 29
activator/inhibitor rules across the 5 genes (see `mined_network_summary.txt`), all at 100%
confidence — e.g. `Gene2 <- Gene1 & !Gene2 & !Gene4` (activator) and `Gene2 <- Gene4` (inhibitor).

Searching for attractors from all 32 initial states (seed 42) finds **4 attractors**: two fixed
points (basins of 5 and 1 states) and two period-3 cycles (empty recorded basins — a quirk that
matches the original R implementation's `searchForAttractors`, which also does not record a basin
for a start state whose trajectory joins an already-discovered attractor's path mid-way through).

## Leukaemia network (`data/Leukeamia_Timeseries.rda`) — parameters and comparison with R

`leukaemia_network_pipeline.py` mines a network from the real leukaemia time series and writes every
parameter it used (`params.json`) plus a ready-to-run R script (`reproduce_in_R.R`) so the same experiment
can be run with the R package and diffed.

```bash
python example/leukaemia_network_pipeline.py --scenario non_temporal   # vs FBM_Leukeamia_Networks.rda
python example/leukaemia_network_pipeline.py --scenario temporal       # vs Leukeamia_Networks.rda
# in R, from the output folder:   Rscript reproduce_in_R.R
python example/leukaemia_network_pipeline.py --scenario non_temporal --compare-csv <folder>/r_rules.csv
diff <folder>/r_network.txt <folder>/python_network.txt    # R's print vs Python's print
```

`python_network.txt` is the mined network printed in R's `print.FundamentalBooleanNetwork` format;
`reproduce_in_R.R` writes `r_network.txt` from R's own `print()`. `reference_<name>.txt` is the stored
R-package network rendered with the same printer, so `diff reference_*.txt python_network.txt` works without running R.

**Data.** 26 samples × 285 genes, 2–3 time points each (0/6/8/24 h), already 0/1, so no discretisation is
applied. All 26 samples are distinct, so `FBNDataReduction` removes none. The `.rda` is read with the
pure-Python `rdata` package (`pyreadr` returns nothing for it).

**Stored reference networks** (the R package does not record the parameters used to make them, so these
are inferred from the rules):

| File(s) | Rules | Timesteps | Inputs / rule | Inferred `temporal` |
|---|---|---|---|---|
| `FBM_Leukeamia_Networks.rda`, `fbm_leukeamia_network.rda` | 2758 | 1 | 1–3 | 1 |
| `Leukeamia_Networks.rda`, `TFBM_Leukeamia_Networks.rda` | 2775 | 1, 2 | 1–3 | 2 |

**Parameters used** (R's defaults, spelled out): `maxK = 3`, `temporal = 1` or `2`,
`threshold_confidence = 1`, `threshold_error = 0`, `threshold_support = 1e-5`, `maxFBNRules = 5`,
all genes as targets and conditional genes. `maxK` is the least certain: R's rules use at most 3 inputs,
but a larger `maxK` could also produce that.

**Result.** The Python network matches R's stored one only partly: 89.6 % of R's rules are identical for the
non-temporal network and 87.5 % for the temporal one (support is equal on every shared rule). Every rule on
*both* sides is valid on the data (confidence 1, support equal to the value recomputed from the data), so
the two are different selections from the same pool of valid rules. Investigated and **ruled out** as the
cause: the support threshold, genes without rules, duplicate-sample removal, the Fisher p-value
implementation, the rule ranking and the per-gene cap. The first level of the tree is where they differ:
R's pool lacks single-gene rules that Python accepts (Fisher p between 0.016 and 0.049).
A partial lead: replacing the Fisher `p <= 0.05` test by an uncorrected Pearson chi-square test with
`p <= 0.0125` reproduces R's decision on all kept rules and raises the match to 91.2 % / 90.9 %, but does not
close the gap, so it is **not** confirmed as the cause. Running `reproduce_in_R.R` and comparing
`r_root_pool.csv` with `python_root_pool.csv` shows exactly which first-level genes differ.
