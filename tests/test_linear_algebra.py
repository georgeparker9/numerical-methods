import pytest
import numpy as np
from numerical_methods import linear_algebra

#Gaussian Elimination Tests
def test_gaussian_elimination_dim():
    with pytest.raises(AssertionError):
        linear_algebra.gaussian_elimination(np.arange(27).reshape(3,3,3))

def test_gaussian_elimination_identity():
     result = linear_algebra.gaussian_elimination(np.identity(3))
     assert (result == np.identity(3)).all()

def test_gaussian_elimination_zeros():
    result = linear_algebra.gaussian_elimination(np.zeros((3,3)))
    assert (result == np.zeros((3,3))).all()

def test_gaussian_elimination_non_trivial():
    result = linear_algebra.gaussian_elimination(np.arange(12).reshape(3,4))
    assert (result == np.array([1,1.25,1.5,1.75,0,1,2,3,0,0,0,0]).reshape(3,4)).all()


# Kernel Basis Tests
def test_kernel_basis_zeros():
    result = linear_algebra.kernel_basis(np.zeros((3,3)))
    assert (result == np.identity(3)).all()

def test_kernel_basis_identity():
    result = linear_algebra.kernel_basis(np.identity(3))
    print(result)
    assert (result == []).all()

def test_kernel_basis_non_trivial():
    result = linear_algebra.kernel_basis(np.arange(12).reshape(3,4))
    print(result)
    assert (result == np.array([1,-2,1,0,2,-3,0,1]).reshape(2,4)).all()