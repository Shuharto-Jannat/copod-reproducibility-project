# COPOD Reproduction Log

## Project

**Paper:** COPOD: Copula-Based Outlier Detection  
**Authors:** Li, Zhao, Botta, Ionescu, and Hu  
**Conference:** IEEE International Conference on Data Mining (ICDM), 2020

## Computing environment

- Platform: GitHub Codespaces
- Operating system: Linux
- Python version: 3.12.1
- Virtual environment: `.venv`
- Original code: included as the Git submodule `original_code/COPOD`

## Initial setup

The official COPOD repository was added as a Git submodule from:

`https://github.com/winstonll/COPOD.git`

A private Python virtual environment was created with:

```bash
python -m venv .venv
```

The required Python packages were installed inside this environment.

## Initial example execution

The example program examined was:

`original_code/COPOD/results/cod_example.py`

It loads the `breastw.mat` dataset and produces feature-level anomaly explanations for observations 69 and 97.

### First issue: execution directory

Running the script from the main project directory produced:

```text
ModuleNotFoundError: No module named 'models'
```

The script assumes that it is executed from the authors' `results` directory.

### Second issue: inconsistent class name

After running the script from the expected directory, it produced:

```text
ImportError: cannot import name 'COD' from 'models.cod'
```

The example imports a class named `COD`, but `models/cod.py` defines the class as `COPOD`.

This appears to be an inconsistency in the released research code.

### Temporary compatibility correction

The example was executed without permanently modifying the authors' code by temporarily replacing:

```python
from models.cod import COD
```

with:

```python
from models.cod import COPOD as COD
```

The command used was:

```bash
sed 's/from models.cod import COD/from models.cod import COPOD as COD/' cod_example.py | PYTHONPATH=.. python -
```

### Result

The example executed successfully and printed feature-level anomaly explanations for observations 69 and 97.

For each observation:

- indices 0–8 represent the dataset's nine features;
- the first set contains that observation's feature-level anomaly scores;
- the `0.9` set contains the 90th-percentile reference scores;
- the `0.99` set contains the 99th-percentile reference scores.

This verifies that the original COPOD implementation can execute in the current environment after addressing the class-name inconsistency. It does not yet reproduce the paper's complete benchmark results.