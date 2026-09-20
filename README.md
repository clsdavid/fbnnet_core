# fbnnet_core

A Python port of [**FBNNet**](https://github.com/clsdavid/FBNNet2_public), an R package implementing the
**Fundamental Boolean Model (FBM)** and **Fundamental Boolean Networks (FBN)** for inferring gene
regulatory networks from time-series (e.g. microarray/RNA-seq) data.

> Chen, L., Kulasiri, D. and Samarasinghe, S. (2018). *A Novel Data-Driven Boolean Model for Genetic
> Regulatory Networks.* Front. Physiol. 9:1328. https://doi.org/10.3389/fphys.2018.01328
>
> Chen, L., Kulasiri, D. and Samarasinghe, S. (2022). *Fundamental Boolean network modelling for
> childhood acute lymphoblastic leukaemia pathways.* Quant. Biol. 10(1):94-121.
> https://doi.org/10.15302/J-QB-021-0280

The performance-critical statistics (gene-probability measures, chi-square/causality tests, and the
recursive cube/tree search) are implemented in C++ (`src/`) and exposed to Python via
[pybind11](https://github.com/pybind/pybind11). The higher-level orchestration, data utilities and
network-mining logic live in the `py_src/` package.

## Project status

The core pipeline described in the original R package (`FBNNet2_public`) has been ported and is
exercised end to end in [`example/reproduce_vignette_experiment.py`](example/reproduce_vignette_experiment.py)
against the same test network the R package's own vignette uses (see
["Reproducing the R vignette experiment"](#reproducing-the-r-vignette-experiment) below).

| Area | R source | Python status |
|---|---|---|
| Time-series / BoolNet-style helpers (`loadNetwork`, `generateBoolNetTimeseries`, `generateAllCombinationBinary`) | `data_utility_FBN.R`, `BoolNet_CellCycle_Network.R` | ✅ Ported (`py_src/boolnet.py`, `py_src/data_utils.py`) |
| General utilities (data reduction, similarity checks, type checks) | `utility_FBN.R` | ✅ Ported (`py_src/general_utils.py`) |
| Orchard Cube construction (gene-probability measures, chi-square/causality tests, recursive tree search) | `cube_FBN.R`, `modelling_FBN.R` | ✅ Ported (`py_src/cube.py` + C++ `src/fbn_core.cpp`, `fbn_tree.cpp`, `fbn_chisq.cpp`) |
| Boolean expression tree parsing / rule construction | `network_utility_FBN.R` | ✅ Ported (`py_src/network_utils.py`) |
| Network mining from a cube (`mineFBNNetwork`) | `network_FBN.R` | ✅ Ported (`py_src/network.py: mine_fbn_network`) |
| Network application/query layer (`loadFBNNetwork`, `mergeNetwork`, `filterNetworkConnections(ByGenes/ByInputGenes)`, `findAllForward/BackwardRelatedGenes`, `find(Forward\|Backward)RelatedNetworkByGenes`, etc.) | `network_application_FBN.R` | ✅ Ported (`py_src/network_app.py`) |
| Attractor search (`searchForAttractors`, `getFBMSuccessor`, `networkFixUpdate`) | `attractor_FBN.R`, parts of `modelling_FBN.R` | ✅ Ported (`py_src/attractor.py`) — faithfully reproduces R's random tie-breaking between competing rules and its basin-tracking quirks (see [example/README.md](example/README.md)) |
| Graph/visualisation (`FBNNetwork.Graph`, `plotNetwork`, attractor drawing) | `graph_FBN.R`, `plot_network_FBN.R` | ✅ Ported as a `networkx` + `matplotlib`-based equivalent (`py_src/network_graph.py`) — see note below |
| Single top-level entry point (`generateFBMNetwork` = discretise + cube + mining in one call) | `application_FBN.R` | ✅ Ported (`py_src/application.py: generate_fbm_network`) |
| `reconstructTimeseries` / `generateSimilaryReport` round-trip accuracy check (referenced in the R vignette) | Not present anywhere in current `FBNNet2_public/R/*.R` sources (only in the vignette/tests, which appear stale relative to the current R package) | ⚠️ Not ported — there is no current R implementation to port from. `py_src/general_utils.py` does have an (independently useful) `check_similarity`/`generate_similar_report` pair, but they don't reproduce this exact R function |

> **Note on visualisation:** `graph_FBN.R` builds `visNetwork`/`igraph`-specific JSON structures for an
> interactive JS widget, which has no direct Python equivalent. `py_src/network_graph.py` instead
> converts an FBN network into a `networkx.MultiDiGraph` and renders it with `matplotlib`. It supports
> both a simplified `gene -> gene` view (`draw_static_network`, `plot_network`) and, via
> `show_rule_nodes=True`, R's actual `gene -> activator/inhibitor rule (timestep) -> gene` 3-tier
> structure with per-input negation highlighting (`to_networkx_graph_with_rules`) — as well as
> attractor drawing (`draw_attractor`). These preserve the same activator/inhibitor/decay styling
> conventions as R but render static images rather than an interactive widget.
>
> **Note on discretisation:** `generate_fbm_network`'s non-boolean-data path uses a small dependency-free
> 1-D 2-means implementation (`py_src/application.py: binarize_time_series`) rather than wrapping
> `BoolNet::binarizeTimeSeries`'s `kmeans`/`edgeDetector`/`scanStatistic` methods — only `"kmeans"`-style
> discretisation is currently supported.

## Repository layout

```
src/            C++ sources for the pybind11 extensions (fbnnet_core, fbnnet_tree,
                fbnnet_utils, fbnnet_matrix, fbnnet_column, fbnnet_chisq)
py_src/         Python package with the higher-level pipeline (named `py_src`, not
                `py`, so it doesn't shadow the third-party `py`/pylib module pytest
                relies on internally)
  boolnet.py      Minimal BoolNet-style .bn network loader
  data_utils.py   Time-series simulation / combination generation helpers
  general_utils.py  Data reduction, similarity, and validation helpers
  cube.py         construct_fbn_cube(): builds the Orchard Cube from time series
  network.py      mine_fbn_network(): mines a FBN network from a cube
  network_app.py  Network loading/merging/filtering/query helpers (forward/backward related genes, etc.)
  network_utils.py  Boolean expression parsing/tree utilities
  attractor.py    searchForAttractors()/getFBMSuccessor() port: FBM attractor search
  network_graph.py  networkx/matplotlib-based network & attractor visualisation
  application.py  generate_fbm_network(): single top-level pipeline entry point
statistic/      Statsmodels/scipy-based statistical tests (e.g. Fisher's exact test)
tests/          unittest test suite exercising the C++ extensions and Python layer
example/        A runnable, end-to-end reproduction of the R package's vignette experiment
                (see "Reproducing the R vignette experiment" below)
example.bn, example_fbn.csv, test_network.txt, test_genes.txt
                Sample network/data files used by the tests and examples
```

## Requirements

* Python >= 3.12
* A C++ compiler toolchain (the build compiles the pybind11 extensions from `src/`)
* See [requirements.txt](requirements.txt): `pybind11`, `numpy`, `statsmodels`, `setuptools`, `networkx`, `matplotlib`
* `scipy` (used by `statistic/chi_test.py`)

## Installation

Use a virtual environment so the build doesn't touch your system Python:

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install .
```

To rebuild from scratch after changing the C++ sources:

```bash
python setup.py clean --all
rm -rf build/ dist/ *.egg-info
pip uninstall -y fbnnet_core
pip install .
```

This builds and installs six extension modules: `fbnnet_core`, `fbnnet_tree`, `fbnnet_utils`,
`fbnnet_matrix`, `fbnnet_column`, and `fbnnet_chisq`.

## Running the tests

```bash
source .venv/bin/activate
python -m unittest discover -s tests -v
```

The project's own Python package lives in `py_src/` (not `py/`) specifically so it doesn't shadow
the third-party `py`/pylib module that some `pytest` internals (`_pytest/compat.py`,
`LEGACY_PATH = py.path.local`) still import at runtime — with a top-level `py/` folder in the repo,
`import py` would resolve to the local package instead and crash pytest with
`AttributeError: module 'py' has no attribute 'path'`. `pytest` also works directly now:

```bash
pytest tests/ -v
```

As of the current codebase, running the full suite gives **124 passed**.

You can run a single test module directly, e.g.:

```bash
python -m unittest tests.test_cube -v
python -m unittest tests.test_fbn_tree -v
```

## Usage

### 1. Load or simulate a Boolean network for training data

```python
from py_src.boolnet import load_network
from py_src.data_utils import generateAllCombinationBinary, generateBoolNetTimeseries
import pandas as pd

network = load_network("example.bn")  # targets/factors .bn file, BoolNet-style

initial_states = generateAllCombinationBinary(network["genes"])
trainingseries = generateBoolNetTimeseries(
    network, initial_states, numMeasurements=43, transition_type="synchronous"
)

# construct_fbn_cube expects a list of pandas DataFrames (genes x time points)
trainingseries = [
    pd.DataFrame(mat, index=network["genes"], columns=[str(j + 1) for j in range(mat.shape[1])])
    for mat in trainingseries
]
```

### 2. Build the Orchard Cube (statistical modelling stage)

```python
from py_src.cube import construct_fbn_cube

genes = list(network["genes"])
cube = construct_fbn_cube(
    target_genes=genes,
    conditional_genes=genes,
    timeseries_cube=trainingseries,
    max_k=5,       # max depth the cube search can dig into
    temporal=1,    # how many previous time steps a target can depend on
    use_parallel=False,
)
```

### 3. Mine a network from the cube

```python
from py_src.network import mine_fbn_network

fbn_network = mine_fbn_network(cube, genes, use_parallel=False)
```

### 4. Or run the whole pipeline in one call

```python
from py_src.application import generate_fbm_network

fbn_network = generate_fbm_network(
    trainingseries,        # a DataFrame or list of DataFrames (genes x timepoints)
    max_k=5,
    max_deep_temporal=1,
    network_only=True,     # set False to also get back {"cube": ..., "network": ...}
)
```

If the input data isn't already boolean, it's discretised first (see the note on discretisation
above).

### 5. Query and filter the network

```python
from py_src.network_app import (
    load_fbn_network,
    find_forward_related_network_by_genes,
    find_backward_related_network_by_genes,
    filter_network_connections_by_genes,
)

fbn_network = load_fbn_network("example_fbn.csv")

# Genes downstream of GeneA (activators only), one level deep
downstream = find_forward_related_network_by_genes(
    fbn_network, target_gene_list=["GeneA"], regulation_type=1, max_deep=1
)

# Genes upstream of GeneA
upstream = find_backward_related_network_by_genes(fbn_network, target_gene_list=["GeneA"], max_deep=1)

# Sub-network restricted to a gene list
subnetwork = filter_network_connections_by_genes(fbn_network, genelist=["GeneA", "GeneB"], exclusive=False)
```

### 6. Search for attractors

```python
from py_src.attractor import search_for_attractors

attractors = search_for_attractors(fbn_network, fbn_network["genes"], transition_type="synchronous")
print(attractors["Attractors"])       # list of attractor cycles (each a list of gene-state dicts)
print(attractors["BasinOfAttractor"]) # states that flow into each attractor
```

### 7. Visualise a network or an attractor

```python
from py_src.network_graph import draw_static_network, plot_network, draw_attractor

draw_static_network(fbn_network)                       # activator=green, inhibitor=red edges
plot_network(fbn_network, target_genes=["GeneA"], direction="forward", max_deep=2)
draw_attractor(attractors, index=0)                    # genes x cycle-step heatmap

# gene(node) -> activator/inhibitor rule (timestep, node) -> gene(node), matching R's structure
draw_static_network(fbn_network, show_rule_nodes=True)
plot_network(fbn_network, target_genes=["GeneA"], direction="forward", max_deep=1, show_rule_nodes=True)
```

## Reproducing the R vignette experiment

The R package's vignette (`FBNNet2_public/vignettes/FBNNet.Rmd`, *"Extract the Fundamental Boolean
Network"*) works through one full, concrete experiment: load a 5-gene BoolNet network
(`ExampleNetwork`), simulate 43 synchronous time steps from every possible initial state, mine a
Fundamental Boolean Network from that data, graph it, and search for its attractors.

[`example/reproduce_vignette_experiment.py`](example/reproduce_vignette_experiment.py) runs that
same experiment with this Python port, against [`example/example.bn`](example/example.bn) — a
byte-for-byte copy of the R package's own `ExampleNetwork` test fixture
(`FBNNet2_public/tests/testthat/example.bn`):

```bash
source .venv/bin/activate
python example/reproduce_vignette_experiment.py
```

This mines 29 activator/inhibitor rules across the 5 genes (all at 100% confidence) and finds 4
FBM attractors (two fixed points, two period-3 cycles) — see
[`example/README.md`](example/README.md) for the full walkthrough, the generated network/attractor
images, and an explanation of which parts of the R vignette (namely, its
`reconstructTimeseries`/`generateSimilaryReport` round-trip check) have no current R implementation
to port from.

## License

MIT — see [LICENSE](LICENSE).
