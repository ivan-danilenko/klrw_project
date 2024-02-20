# all .pyx files in a folder
from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "solver",
        ["solver.pyx"],
        include_dirs=[numpy.get_include()]
    ),
]
setup(
    name="klrw.solver",
    ext_modules=cythonize(extensions),
)
