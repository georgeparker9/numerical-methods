import pytest
import numpy as np

# Import the functions to be tested
from numerical_methods import differential_equations

# Helper Function Tests
def test_initialise_array_scalar():
    # Test with scalar input
    assert differential_equations._initialise_array(5) == 5.0
    assert differential_equations._initialise_array(3.14) == 3.14

def test_initialise_array_array():
    # Test with array-like input
    y_0 = [1, 2, 3]
    result = differential_equations._initialise_array(y_0)
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, np.array([1.0, 2.0, 3.0]))

# Euler's Method Tests
def test_euler_method_scalar():
    # Test with a simple ODE: dy/dx = x, y(0) = 0
    f = lambda x, y: x
    x_0, y_0, h, n = 0, 0, 0.01, 10
    sol = differential_equations.euler_method(f, x_0, y_0, h, n)

    # Exact solution: y = x^2 / 2
    for x, y in sol.items():
        expected_y = x**2 / 2
        assert abs(y - expected_y) < 1e-2

def test_euler_method_system():
    # Test with a system of ODEs: dy/dx = [y[1], -y[0]], y(0) = [0, 1]
    f = lambda x, y: np.array([y[1], -y[0]])
    x_0, y_0, h, n = 0, np.array([0, 1]), 0.01, 100
    sol = differential_equations.euler_method(f, x_0, y_0, h, n)

    # Exact solution: y = [sin(x), cos(x)]
    for x, y in sol.items():
        expected_y = np.array([np.sin(x), np.cos(x)])
        print(y,expected_y)
        assert np.allclose(y, expected_y, atol=1e-2)

# Adams Bashforth Tests
def test_adams_bashforth_scalar():
    # Test with a simple ODE: dy/dx = x, y(0) = 0
    f = lambda x, y: x
    x_0, y_0, h, n = 0, 0, 0.01, 10
    sol = differential_equations.adams_bashforth(f, x_0, y_0, h, n)

    # Exact solution: y = x^2 / 2
    for x, y in sol.items():
        expected_y = x**2 / 2
        assert abs(y - expected_y) < 1e-2

def test_adams_bashforth_system():
    # Test with a system of ODEs: dy/dx = [y[1], -y[0]], y(0) = [0, 1]
    f = lambda x, y: np.array([y[1], -y[0]])
    x_0, y_0, h, n = 0, np.array([0, 1]), 0.01, 100
    sol = differential_equations.adams_bashforth(f, x_0, y_0, h, n)

    # Exact solution: y = [sin(x), cos(x)]
    for x, y in sol.items():
        expected_y = np.array([np.sin(x), np.cos(x)])
        assert np.allclose(y, expected_y, atol=1e-2)

# Test Runge-Kutta method
def test_runge_kutta_scalar():
    # Test with a simple ODE: dy/dx = x, y(0) = 0
    f = lambda x, y: x
    x_0, y_0, h, n = 0, 0, 0.01, 10
    sol = differential_equations.runge_kutta(f, x_0, y_0, h, n)

    # Exact solution: y = x^2 / 2
    for x, y in sol.items():
        expected_y = x**2 / 2
        assert abs(y - expected_y) < 1e-2

def test_runge_kutta_system():
    # Test with a system of ODEs: dy/dx = [y[1], -y[0]], y(0) = [0, 1]
    f = lambda x, y: np.array([y[1], -y[0]])
    x_0, y_0, h, n = 0, np.array([0, 1]), 0.01, 100
    sol = differential_equations.runge_kutta(f, x_0, y_0, h, n)

    # Exact solution: y = [sin(x), cos(x)]
    for x, y in sol.items():
        expected_y = np.array([np.sin(x), np.cos(x)])
        assert np.allclose(y, expected_y, atol=1e-2)

# Test invalid step size
def test_invalid_step_size():
    f = lambda x, y: x
    x_0, y_0, h, n = 0, 0, -0.1, 10

    with pytest.raises(AssertionError):
        differential_equations.euler_method(f, x_0, y_0, h, n)

    with pytest.raises(AssertionError):
        differential_equations.adams_bashforth(f, x_0, y_0, h, n)

    with pytest.raises(AssertionError):
        differential_equations.runge_kutta(f, x_0, y_0, h, n)