import pytest
from math import sin

from numerical_methods import root_finding

#Binary Search Tests
def test_binary_search_linear():
    f = lambda x: x - 2
    root = root_finding.binary_search(f, 1, 3, 1e-5)
    assert abs(root - 2) < 1e-5

def test_binary_search_quadratic():
    # Test with a quadratic function
    f = lambda x: x**2 - 4
    root = root_finding.binary_search(f, 1, 5, 1e-5)
    assert abs(root - 2) < 1e-5

def test_binary_search_trig():
    # Test with a trigonometric function
    f = lambda x: sin(x)
    root = root_finding.binary_search(f, 2, 4, 1e-5)
    assert abs(root - 3.141592653589793) < 1e-5

def invalid_interval_test():
    # Test an invalid interval
    f = lambda x: x**2
    with pytest.raises(AssertionError):
        root_finding.binary_search(f, 1, 2, 1e-5)

# Fixed Point Iteration Test
def test_fixed_point_iteration_rational():
    # Test with a simple function
    f = lambda x: (x + 2 / x) / 2
    fixed_point = root_finding.fixed_point_iteration(f, 1.0, 1e-5, 1000)
    assert abs(fixed_point - 1.414213562373095) < 1e-5

def test_fixed_point_iteration_diverge():
    # Test with a function that doesn't converge
    f = lambda x: x + 1
    fixed_point = root_finding.fixed_point_iteration(f, 1.0, 1e-5, 10)
    assert abs(fixed_point - 11) < 1e-5  # Should reach N_max iterations

# Central Difference Tests
def test_central_difference_linear():
    # Test with a simple linear function
    f = lambda x: 2 * x + 3
    derivative = root_finding.central_difference(f, 2, 1e-5)
    assert abs(derivative - 2) < 1e-5

def test_central_difference_quad():
    # Test with a quadratic function
    f = lambda x: x**2
    derivative = root_finding.central_difference(f, 2, 1e-5)
    assert abs(derivative - 4) < 1e-5

def test_central_difference_trig():
    # Test with a trigonometric function
    f = lambda x: sin(x)
    derivative = root_finding.central_difference(f, 0, 1e-5)
    assert abs(derivative - 1) < 1e-5

# Newton Raphson Tests
def test_newton_raphson_linear():
    # Test with a simple linear function
    f = lambda x: x - 2
    root = root_finding.newton_raphson(f, 1.0, 1e-5, 1000)
    assert abs(root - 2) < 1e-5

def test_newton_raphson_quad():
    # Test with a quadratic function
    f = lambda x: x**2 - 4
    root = root_finding.newton_raphson(f, 3.0, 1e-5, 1000)
    assert abs(root - 2) < 1e-5

def test_newton_raphson_trig():
    # Test with a trigonometric function
    f = lambda x: sin(x)
    root = root_finding.newton_raphson(f, 3.0, 1e-5, 1000)
    assert abs(root - 3.141592653589793) < 1e-5

def test_newton_raphson_diverge():
    # Test with a function that doesn't converge
    f = lambda x: x**2 + 1
    
    with pytest.raises(ZeroDivisionError):
        root_finding.newton_raphson(f, 1.0, 1e-5, 10)