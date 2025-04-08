# setup.py
from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "fbnnet_core",
        ["src/fbn_utils.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++"
    )
]

setup(
    name="fbnnet_core",
    ext_modules=ext_modules,
    install_requires=["pybind11>=2.13", "statsmodels"], 
)
