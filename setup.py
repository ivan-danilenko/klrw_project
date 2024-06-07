# all .pyx files in a folder
from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

import os
from pathlib import Path
# directory of setup.py
dir_path = os.path.dirname(os.path.realpath(__file__))


PACKAGE_NAME = "klrw.cython_exts"
PACKAGE_PATH = dir_path + "/klrw/cython_exts"
PICKLES_DIR = dir_path + "/pickles/"

extensions = [
    Extension(
        "{}.sparse_csc".format(PACKAGE_NAME),
        sources=["{}/sparse_csc.pyx".format(PACKAGE_PATH)],
        include_dirs=[numpy.get_include()],
    ),
    Extension(
        "{}.sparse_csr".format(PACKAGE_NAME),
        sources=["{}/sparse_csr.pyx".format(PACKAGE_PATH)],
        include_dirs=[numpy.get_include()],
    ),
    Extension(
        "{}.sparse_addition".format(PACKAGE_NAME),
        sources=["{}/sparse_addition.pyx".format(PACKAGE_PATH)],
        include_dirs=[numpy.get_include()],
    ),
    Extension(
        "{}.perfect_complex_corrections_ext".format(PACKAGE_NAME),
        sources=["{}/perfect_complex_corrections_ext.pyx".format(PACKAGE_PATH)],
        include_dirs=[numpy.get_include()],
    ),
    Extension(
        "{}.combinatorial_ebranes_ext".format(PACKAGE_NAME),
        sources=["{}/combinatorial_ebranes_ext.pyx".format(PACKAGE_PATH)],
        include_dirs=[numpy.get_include()],
    ),
    Extension(
        "{}.sparse_multiplication".format(PACKAGE_NAME),
        sources=["{}/sparse_multiplication.pyx".format(PACKAGE_PATH)],
        include_dirs=[numpy.get_include()],
    ),
]

if __name__ == "__main__":
    setup(
        name=PACKAGE_NAME,
        #    package_dir={'cython_ext': '/cython'},
        #    include_dirs=["."],
        ext_modules=cythonize([e for e in extensions]),
        packages=[PACKAGE_NAME],
        package_dir={PACKAGE_NAME: PACKAGE_PATH},
    )
    
    # Additionally make the pickles directory if it doesn't exist
    Path(PICKLES_DIR).mkdir(parents=True, exist_ok=True)
