# Changelog

## Unreleased

- `use_parallel` works with the `spawn` start method (Windows) instead of silently running on one core: new `fbnnet_core._parallel` picks `fork` when available, else `spawn`, and workers rebuild their state from the time series. Covers the cube, the rule mining and `reconstruct_timeseries`. `FBNNET_START_METHOD=fork|spawn` overrides the choice (CI runs the parallel tests with `spawn`). Scripts must guard the entry point with `if __name__ == "__main__":` on those platforms. The private `_FORK_AVAILABLE` flags were removed.
- `use_parallel` now also applies to rule mining: `mine_fbn_network` / `search_fbn_core` (and so `generate_fbm_network`) mine the target genes in forked workers and accept `max_workers` / `chunksize`. The rule search and the per-gene removal of subsumed rules both run in the workers (`mine_fbn_network_with_cores` / `mine_fbn_network_stage2` take the same options). The result is identical; mining a 2,500-gene synthetic cube went from 20.3 s to 9.8 s with 8 workers (the rule search alone from 21.9 s to 6.3 s at 3,000 genes). The "use_parallel is not supported" warning is gone.
- **Default backend changed**: `construct_fbn_cube` and `generate_fbm_network` now take `backend="auto"`, which selects `"tensor"` (identical cubes; never slower in the benchmarks, 6-14x faster beyond a few hundred genes). Pass `backend="recursive"` for the previous behaviour. Note that the tensor backend is AGPL-licensed (see README, "Copyright and licensing").
- `backend="tensor"` evaluates each tree node columnar (new `fbnnet_core.tensor_nodes`: shared float32 indicator matrices built once before forking, one matrix product per step, measures computed once per distinct count tuple) and the recursive tree no longer rebuilds the unprocessed-gene lists at every gene; results are identical (`gene_probabilities_measurements_reference` keeps the old implementation for differential tests).
- New `tools/benchmark_cube.py`: benchmark harness for `construct_fbn_cube` (sequential/parallel x recursive/tensor on random targets, parity check, survivor statistics, per-target time distribution, cost-model fits and a full-run extrapolation; synthetic, bundled, `.npz` and CEL data).
- `download_hgu133plus2_annotation` fetches `hgu133plus2.sqlite` and `org.Hs.eg.sqlite` from the Bioconductor packages (no R, no conversion); README documents the manual download. The annotation is not bundled because of its size.
- `load_hgu133plus2_annotation` reads the SQLite files of the Bioconductor packages `hgu133plus2.db` and `org.Hs.eg.db` (no R needed); a path to `hgu133plus2.sqlite`, or the environment variable `FBNNET_HGU133PLUS2_DB`, can be used wherever an annotation is accepted.
- New `fbnnet_core.bioinformatics` module with the tools of the R `FBNBioinformatics.R` / `sample_FBN.R`: `convert_affy_raw_data_into_normalized_structure_data`, `get_related_affy_raw_data`, `normalize_timeseries_raw_data`, `convert_into_sample_timeseries`, `reorder_sample_timeseries`, `identify_differentially_expressed_genes`, `get_specific_expressed_genes`, `convert_timeseries_probeset_name_to_gene_name`, `map_probeset_names`, `map_to_gene_name_with_hgu133plus2_db`, `convert_probeset_to_gene` and `dissolve`.
- New `fbnnet_core.geo` module: `download_genome_data_exprs` (R `downloadGenomeDataExprs`, series matrix from NCBI GEO) and `download_geo_raw_files` (fetch and unpack a series' `*_RAW.tar`). New `map_probeset_names` (R `mapProbesetNames`). README states that CEL files are not shipped and where to download them.
- New `fbnnet_core.cel_import` module (and `python -m fbnnet_core.cel_import`): converts raw Affymetrix CEL files plus a text CDF into time series for `generate_fbm_network` / `construct_fbn_cube`, following the R `FBNBioinformatics` pipeline (quantile normalisation + median polish, sample grouping by file name, differentially-expressed probe set selection, probe set to gene mapping with the bundled DAVID list, binarisation).
- `mine_fbn_network(use_parallel=True)` now emits a `UserWarning` (rule mining is single-process; parallelism applies to cube construction). `generate_fbm_network` no longer triggers it.
- README documents `max_workers`, `chunksize` and `backend`.
- New tests check that parallel runs and the `tensor` backend give results identical to the sequential `recursive` engine.

## v1.1.7

Dependency maintenance.

- `pillow>=13.0` is now required on Python 3.11+ (pulled in by matplotlib); earlier versions have known CVEs. Pillow 13 does not support Python 3.10, so 3.10 installs may still resolve to a vulnerable Pillow 12.x.
- The build requires `setuptools>=84.1` (fixes GHSA-xmj3-9gf7-h52w).
- Release workflows use the current GitHub Actions majors (`checkout` v7, `setup-python` v7, `upload-artifact` v7, `download-artifact` v8) and build wheels with `cibuildwheel` 4.3.
- Verified against numpy 2.5, pandas 3.1, scipy 1.18, networkx 3.7, matplotlib 3.11, pybind11 3.1.

## v1.1.6

Licensing.

- Code ported from the R package stays MIT (the copyright notice now includes the R package's).
- The new tensor mining backend (`tensor_mining.py`, `batched_counts.py`) is AGPL-3.0-or-later; see `LICENSE`.
- Earlier tags (up to v1.1.5) were published entirely under MIT.

## v1.1.5

Release pipeline fixes.

- Fisher exact p-value: a result that is mathematically 0.05 is no longer rounded above the cutoff on Apple ARM.
- Wheels are built with `cibuildwheel` 2.23 (the pinned `packaging` of 2.21 broke `setuptools>=77`).
- Linux wheels are built on `manylinux_2_28`, so the test environment installs pandas and pillow from wheels.

## v1.1.4

R-faithful network graphs, bundled data sets and a rewritten README.

- `plot_network` accepts the R plot types (`static`, `staticSlice`, `dynamic`, `forward_1a`-`forward_4b`, `backward_1a`-`backward_2b`) and returns an `FBNGraph`.
- Graphs reproduce the R colours, shapes, legends and titles; draw them with matplotlib (`graph.draw()`, `graph.save("x.png")`) or export interactive HTML (`graph.save("x.html")`, shown inline in Jupyter).
- New `convert_to_network_graphic_object`, `convert_to_ngo`, `fbn_network_graph`, `draw_dynamic_for_one_matrix`; `draw_attractor(network, attractors, index)` now draws the attractor as a layered network (the old heatmap is `plot_attractor_heatmap`).
- New `merge_cluster_networks`.
- New `load_dataset` / `available_datasets` with all R example data (leukaemia networks and time series, cell-cycle, DAVID annotation, ...).

## v1.1.1

Enhanced FBN Graph presentation.

## v1.1.0

Initial PyPI distribution of the Python FBNNet port.

- Install as `fbnnet-core`; import the public API from `fbnnet_core`.
- Bundle the Python implementation and C++17/pybind11 extensions in one namespaced package.
- Include network mining, cube construction, attractor analysis, visualization, and BoolNet-style discretization.
- Build Linux and macOS wheels for CPython 3.10-3.13; source builds require a C++17 compiler.

This packaging release changes internal imports: the former `py_src.*` modules and top-level `fbnnet_*` extension modules are now under `fbnnet_core` (compiled modules are private submodules).
