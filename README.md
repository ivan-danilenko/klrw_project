# KLRW Algebras

The project implements (planar) Khovanov-Lauda-Rouquier-Webster (KLRW) algebras, constructed in [[KhL](https://arxiv.org/abs/0804.2080),[R](https://arxiv.org/abs/0812.5023),[W](https://arxiv.org/abs/1309.3796)]. It also illustrates its applications in link invariants. The main interest of the creator was to allow to computations for Coulomb branches, namely working with Fukaya category \[[ADLSZ](https://arxiv.org/abs/2406.04258)\] and derived category of coherent sheaves \[[W1](https://arxiv.org/abs/1905.04623), [W2](https://arxiv.org/abs/2211.02099)\].

## Getting Started

To run the project you might need to upgrade/install packages and compile C-extensions written in Cython.

### Prerequisites

The project runs on [SageMath](https://www.sagemath.org/), an open-source mathematics software system with Python-based syntax.  
You can find the installation guide for SageMath [here](https://doc.sagemath.org/html/en/installation/index.html).  
The following packages are required:

```
cython version >= 3.0  
scipy version >=1.11  
numpy version >= 1.23  
gurobipy version >= 10.0
```

SageMath comes with versions of [Cython](https://cython.org/), [SciPy](https://scipy.org/), and [NumPy](https://numpy.org/), though they might need an update. The user can use the following terminal command 

```
sage -pip install --upgrade <package name>
```

or execute

```
%pip install --upgrade <package name>
```

in any cell of any Jupyter (.ipynb) notebook with **Sage** kernel running.

[Gurobi](https://www.gurobi.com/) is a solver developed by Gurobi Optimization, LLC, and it is possible to obtain a [free academic licence](https://www.gurobi.com/features/academic-named-user-license/). User can install `gurobipy` by

```
sage -pip install gurobipy
```

or executing

```
%pip install gurobipy
```

in any cell of any Jupyter (.ipynb) notebook with **Sage** kernel running.  
Large systems are supported only in licenced copies of Gurobi. After getting a licence [e.g. [free academic]()] follow the steps on Gurobi official website, e.g. [here](https://support.gurobi.com/hc/en-us/articles/360059842732-How-do-I-set-up-a-license-without-installing-the-full-Gurobi-package).

### Installing

To compile the Cython extensions, run

```
sage setup.py build_ext --inplace
```
from the folder of the project, or run

```
%run setup.py build_ext --inplace
```

in any cell of any Jupyter (.ipynb) notebook in the folder of the project with **Sage** kernel running.

## Examples

* `testing_combinatorial_ebrane.ipynb` shows how to compute links from a braid in bridge representation,
* `swipe_move_A1.ipynb` shows how to prove the "swipe move" isomorphism, see appendix A in [[ALR](https://arxiv.org/abs/2305.13480)],
* `cobordism_example.ipynb` shows the simplest example of cobordims actions on E-branes.

## Tests

Jupyter notebook `klrw_test.ipynb` tests the implementation of KLRW algebra: relations, etc.

## Authors

* **Ivan Danilenko** - *Main part of the implementation in Sage* - [ivan-danilenko](https://github.com/ivan-danilenko)

with contributions from

* **Elise LePage** - [eliselepage](https://github.com/eliselepage)
* **Marco David** - [marco-david](https://github.com/marco-david)

Originally the project was inspired by the Mathematica code written for [[ALR](https://arxiv.org/abs/2305.13480)].

## License

This project is licensed under the GNU General Public License, version 3. See the [COPYING.txt](COPYING.txt) file for details

## Acknowledgments

The authors wanted to thank the people following people who haven't contributed by writing the code [yet?], but where extremely helpful with discussions on the project:

* Mina Aganagic
* Yixuan Li
* Miroslav Rapcak
* Vivek Shende
* Peng Zhou

Special thanks to the creators of [SageMath](https://www.sagemath.org/).