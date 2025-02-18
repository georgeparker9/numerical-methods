from typing import Callable, Dict
import numpy as np
import numpy.typing as npt

def _initialise_array(y_0: npt.ArrayLike) -> np.ndarray | float:
    """
    Helper Function.
    Initializes an array from the input `y_0`. If `y_0` is a scalar, it returns it as a float.
    Otherwise, it converts `y_0` to a NumPy array.
    """
    if isinstance(y_0, (float, int)):
        return y_0
    return np.array(y_0, dtype=np.float64)


def euler_method(
    f: Callable, x_0: float, y_0: npt.ArrayLike, h: float, n: int
) -> Dict[float, float | np.ndarray]:
    """
    Approximates the solution to a differential equation using Euler's method.

    Args:
        f: The derivative function.
        x_0: The initial x-value.
        y_0: The initial y-value(s).
        h: The step size.
        n: The number of steps.

    Returns:
        A dictionary mapping x-values to their corresponding y-values.
    """

    assert h > 0, "h must have a positive value."

    x, y = x_0, _initialise_array(y_0)
    sol = {x: np.copy(y)}
    for _ in range(n):
        y += h * f(x, y)
        x += h
        sol[x] = np.copy(y)
        print(sol)

    return sol


def adams_bashforth(
    f: Callable, x_0: float, y_0: npt.ArrayLike, h: float, n: int
) -> Dict[float, float | np.ndarray]:
    """
    Approximates the solution to a differential equation using the Adams-Bashforth method.

    Args:
        f: The derivative function.
        x_0: The initial x-value.
        y_0: The initial y-value(s).
        h: The step size.
        n: The number of steps.

    Returns:
        A dictionary mapping x-values to their corresponding y-values.
    """

    assert h > 0, "h must have a positive value."

    x, y = x_0, _initialise_array(y_0)
    sol = {x: np.copy(y)}

    y += h * f(x, y)
    x += h
    sol[x] = np.copy(y)
    x_temp = x_0
    for _ in range(n - 1):
        y += h * ((3 / 2) * f(x, sol[x]) - (1 / 2) * f(x - h, sol[x_temp]))
        x_temp = x
        x += h
        sol[x] = np.copy(y)
    return sol


def runge_kutta(
    f: Callable, x_0: float, y_0: npt.ArrayLike, h: float, n: int
) -> Dict[float, float | np.ndarray]:
    """
    Approximates the solution to a differential equation using the Runge-Kutta method.

    Args:
        f: The derivative function.
        x_0: The initial x-value.
        y_0: The initial y-value(s).
        h: The step size.
        n: The number of steps.

    Returns:
        A dictionary mapping x-values to their corresponding y-values.
    """

    assert h > 0, "h must have a positive value."

    x, y = x_0, _initialise_array(y_0)
    sol = {x: np.copy(y)}

    for _ in range(n):
        k1 = f(x, y)
        k2 = f(x + h / 2, y + h * k1 / 2)
        k3 = f(x + h / 2, y + h * k2 / 2)
        k4 = f(x + h, y + h * k3)

        y += (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
        x += h
        sol[x] = np.copy(y)

    return sol

f = lambda x, y: np.array([y[1], -y[0]])
x_0, y_0, h, n = 0, np.array([0, 1]), 0.01, 100
sol = runge_kutta(f, x_0, y_0, h, n)

# Exact solution: y = [sin(x), cos(x)]
for x, y in sol.items():
    expected_y = np.array([np.sin(x), np.cos(x)])
    # assert np.allclose(y, expected_y, atol=2e-2)
    print(x,y,expected_y)