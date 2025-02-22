# Numerical Methods

This is a python program that implements various numerical methods in root finding, differential equation solving, and numerical linear algebra. There are plans to expand the library of available functions to improve utility.

### Documentation
To see documentation, read `docs.md`. 

## Using the code
Install dependencies
```
pip install -r requirements.txt
```
Run the code
```
python root_finding.py
python differential_equations.py
python linear_algebra.py
```
## Installing with Poetry
- Ensure [poetry](https://python-poetry.org/) is installed on your machine.
- Once the repository is cloned, navigate to it's path.
- run `poetry install` to install the module
- run `source .venv/bin/activate` on mac/linux or `& .venv\Scripts\activate` on windows to activate the virtual environment.
- you can now use `import numerical-methods` in your python script.

If you want to use the module globally, run the following outside of poetry's virtual environment
```
poetry build
pip install dist/numerical-methods-*.whl
```
