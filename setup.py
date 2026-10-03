"""Builds the C++ extensions; all other metadata lives in pyproject.toml."""
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

# Each extension is a separate module of the fbnnet_core package (fbnnet_core._core, ...).
_EXTENSIONS = {
    "_tree": ["fbn_tree.cpp", "fbn_core.cpp", "fbn_utils.cpp", "fbn_matrix.cpp", "fbn_matrix_bindings.cpp",
              "fbn_column.cpp", "fbn_column_bindings.cpp", "fbn_chisq.cpp"],
    "_core": ["fbn_core.cpp", "fbn_utils.cpp", "fbn_matrix.cpp", "fbn_matrix_bindings.cpp",
              "fbn_column.cpp", "fbn_column_bindings.cpp", "fbn_chisq.cpp"],
    "_utils": ["fbn_utils.cpp", "fbn_chisq.cpp"],
    "_matrix": ["fbn_matrix.cpp", "fbn_matrix_bindings.cpp"],
    "_column": ["fbn_column.cpp", "fbn_column_bindings.cpp"],
    "_chisq": ["fbn_chisq.cpp"],
}

ext_modules = [
    Pybind11Extension(
        f"fbnnet_core.{name}",
        sorted(f"src/{source}" for source in sources),
        include_dirs=["src"],
        cxx_std=17,
    )
    for name, sources in _EXTENSIONS.items()
]

setup(ext_modules=ext_modules, cmdclass={"build_ext": build_ext})
