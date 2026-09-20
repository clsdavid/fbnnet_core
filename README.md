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

This is a work in progress. Compared with the original R package (`FBNNet2_public`), the **cube
construction / statistical modelling stage is implemented and tested**, but the **network
construction / application layer is not yet complete**:

| Area | R source | Python status |
|---|---|---|
| Time-series / BoolNet-style helpers (`loadNetwork`, `generateBoolNetTimeseries`, `generateAllCombinationBinary`) | `data_utility_FBN.R`, `BoolNet_CellCycle_Network.R` | ✅ Ported (`py_src/boolnet.py`, `py_src/data_utils.py`) |
| General utilities (data reduction, similarity checks, type checks) | `utility_FBN.R` | ✅ Ported (`py_src/general_utils.py`) |
| Orchard Cube construction (gene-probability measures, chi-square/causality tests, recursive tree search) | `cube_FBN.R`, `modelling_FBN.R` | ✅ Ported (`py_src/cube.py` + C++ `src/fbn_core.cpp`, `fbn_tree.cpp`, `fbn_chisq.cpp`) |
| Boolean expression tree parsing / rule construction | `network_utility_FBN.R` | ✅ Ported (`py_src/network_utils.py`) |
| Network mining from a cube (`mineFBNNetwork`) | `network_FBN.R` | ⚠️ Ported (`py_src/network.py: mine_fbn_network`) but **incomplete** — `tests/test_network.py` currently fails an assertion because the mined network does not yet cover all input genes |
| Network application/query layer (`findForwardRelatedNetworkByGenes`, `findAllBackwardRelatedGenes`, `filterNetworkConnectionsByGenes`, etc.) | `network_application_FBN.R` | ❌ Not ported — only `load_fbn_network`, `convert_to_boolean_network_collection`, `merge_network`, `filter_network_connections` exist in `py_src/network_app.py` |
| Attractor search (`searchForAttractors`) | `attractor_FBN.R` | ❌ Not ported |
| Graph/visualisation (`FBNNetwork.Graph`, `plotNetwork`) | `graph_FBN.R`, `plot_network_FBN.R` | ❌ Not ported |
| Single top-level entry point (`generateFBMNetwork` = cube + mining in one call) | `application_FBN.R` | ❌ No Python equivalent yet — call `construct_fbn_cube()` then `mine_fbn_network()` manually |

In short: the **modelling** half of the pipeline (turning time-series data into a statistical "Orchard
Cube" of candidate activator/inhibitor rules) is done. The **network** half (mining a clean final
network from that cube, querying/filtering it, finding attractors, and visualising it) still needs work.

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
  network_app.py  Network loading/merging/filtering helpers
  network_utils.py  Boolean expression parsing/tree utilities
statistic/      Statsmodels/scipy-based statistical tests (e.g. Fisher's exact test)
tests/          unittest test suite exercising the C++ extensions and Python layer
example.bn, example_fbn.csv, test_network.txt, test_genes.txt
                Sample network/data files used by the tests and examples
```

## Requirements

* Python >= 3.12
* A C++ compiler toolchain (the build compiles the pybind11 extensions from `src/`)
* See [requirements.txt](requirements.txt): `pybind11`, `numpy`, `statsmodels`, `setuptools`
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

As of the current codebase, running the full suite gives **65 passed, 1 failure, 1 error**:

* `tests/test_chisq.py` errors with `ModuleNotFoundError: No module named 'fbn_chisq'` — the test
  imports the wrong module name (the built extension is `fbnnet_chisq`, per `setup.py`).
* `tests/test_network.py::test_network` fails because `mine_fbn_network()` does not yet return a
  network that covers all input genes — this is the incomplete "network" stage described above.

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

### 3. Mine a network from the cube (network stage — experimental/incomplete)

```python
from py_src.network import mine_fbn_network

fbn_network = mine_fbn_network(cube, genes, use_parallel=False)
```

Note: as described above, this step is not yet fully reliable — always validate the resulting
`fbn_network['genes']` / rules against your expectations before relying on it.

## License

MIT — see [LICENSE](LICENSE).
