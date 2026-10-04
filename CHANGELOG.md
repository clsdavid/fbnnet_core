# Changelog

## v1.1.1

Enhanced FBN Graph presentation.

## v1.1.0

Initial PyPI distribution of the Python FBNNet port.

- Install as `fbnnet-core`; import the public API from `fbnnet_core`.
- Bundle the Python implementation and C++17/pybind11 extensions in one namespaced package.
- Include network mining, cube construction, attractor analysis, visualization, and BoolNet-style discretization.
- Build Linux and macOS wheels for CPython 3.10-3.13; source builds require a C++17 compiler.

This packaging release changes internal imports: the former `py_src.*` modules and top-level `fbnnet_*` extension modules are now under `fbnnet_core` (compiled modules are private submodules).
