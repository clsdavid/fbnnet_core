# setup.py
from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "fbnnet_tree",
        ["src/fbn_tree.cpp", "src/fbn_core.cpp", "src/fbn_utils.cpp", "src/fbn_matrix.cpp", "src/fbn_matrix_bindings.cpp", "src/fbn_column.cpp", "src/fbn_column_bindings.cpp", "src/fbn_chisq.cpp"],
        include_dirs=[pybind11.get_include(), "src"],
        language="c++"
    ),
    Extension(
        "fbnnet_core",
        ["src/fbn_core.cpp", "src/fbn_utils.cpp", "src/fbn_matrix.cpp", "src/fbn_matrix_bindings.cpp", "src/fbn_column.cpp", "src/fbn_column_bindings.cpp", "src/fbn_chisq.cpp"],
        include_dirs=[pybind11.get_include(), "src"],
        language="c++"
    ),
    Extension(
        "fbnnet_utils",
        ["src/fbn_utils.cpp", "src/fbn_chisq.cpp"],
        include_dirs=[pybind11.get_include(), "src"],
        language="c++"
    ),
    Extension(
        "fbnnet_matrix",
        ["src/fbn_matrix.cpp", "src/fbn_matrix_bindings.cpp"],
        include_dirs=[pybind11.get_include(), "src"],
        language="c++"
    ),
    Extension(
        "fbnnet_column",
        ["src/fbn_column.cpp", "src/fbn_column_bindings.cpp"],
        include_dirs=[pybind11.get_include(), "src"],
        language="c++"
    ),
    Extension(
        "fbnnet_chisq",
        ["src/fbn_chisq.cpp"],
        include_dirs=[pybind11.get_include(), "src"],
        language="c++"
    ),
]

setup(
    name="fbnnet_core",
    ext_modules=ext_modules,
    install_requires=["pybind11>=2.13", "statsmodels"], 
)
