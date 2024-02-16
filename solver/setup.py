# all .pyx files in a folder
from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("Solver", ["Solver.pyx"], include_dirs=[numpy.get_include()]),
]
setup(
    name="Solver",
    ext_modules=cythonize(extensions),
)
