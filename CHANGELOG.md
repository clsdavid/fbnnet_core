# Changelog

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
