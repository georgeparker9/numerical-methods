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

### Root Finding
Results of binary_search(lambda x: x\*\*2-2\*x+1,-10,10,1e-5):
```
-1.618037223815918
```
Results of fixed_point_iteration(lambda x: (x\*\*2)/3+2/3,1,1e-5,100):
```
1.0
```
Results of newton_raphson(lambda x: -x\*\*3+5\*x\*\*2-10\*x+5,2,1e-5,100):
```
0.7243177963489936
```