from typing import Callable


def binary_search(f: Callable, a: float, b: float, epsilon: float) -> float:
    """
    Finds an approximation for a root of the function `f` within the interval [a, b] using the bisection method.

    Args:
        f: The function whose root is to be found.
        a, b: The endpoints of the interval. f(a) and f(b) must have opposite signs.
        epsilon: The tolerance for stopping the search.

    Returns:
        The approximate root of the function `f`.
    """

    
    assert(
        f(a) * f(b) <= 0
    ), "given function evaluates to the same parity at the endpoints."

    assert epsilon > 0, "epsilon must have a positive value."

    lower, upper = min(a, b), max(a, b)

    while upper - lower >= epsilon:
        midpoint = (upper + lower) / 2
        if f(upper) * f(midpoint) > 0:
            upper = midpoint
        elif f(lower) * f(midpoint) > 0:
            lower = midpoint
        else:
            return midpoint
    return (upper + lower) / 2


def fixed_point_iteration(f: Callable, x_0: float, epsilon: float, N_max: int) -> float:
    """
    Finds an approximation for a fixed point of the function `f` using fixed-point iteration.

    Args:
        f: The function for which a fixed point is to be found.
        x_0: The initial estimate.
        epsilon: The tolerance for stopping the iteration.
        N_max: The maximum number of iterations.

    Returns:
        The approximate fixed point of the function `f`.
    """

    assert epsilon > 0, "epsilon must have a positive value."

    x = x_0
    for _ in range(N_max):
        y = f(x)
        if abs(y - x) < epsilon:
            return y
        x = y
    return x


def central_difference(f: Callable, x: float, epsilon: float) -> float:
    """
    Finds an approximation for the derivative of the function `f` at `x` using the central difference method.

    Args:
        f: The function to differentiate.
        x: The point at which to compute the derivative.
        epsilon: The step size for the central difference.

    Returns:
        The approximate derivative of `f` at `x`.
    """

    assert epsilon > 0, "epsilon must have a positive value."

    return (f(x + epsilon) - f(x - epsilon)) / (2 * epsilon)


def newton_raphson(
    f: Callable, x_0: float, epsilon: float, N_max: int, h: float = 1e-5
) -> float:
    """
    Finds an approximation of the root of the function `f` using the Newton-Raphson method.

    Args:
        f: The function whose root is to be found.
        x_0: The initial guess.
        epsilon: The tolerance for stopping the iteration.
        N_max: The maximum number of iterations.
        h: The step size for numerical differentiation (default: 1e-5).

    Returns:
        The approximate root of the function `f`.
    """

    assert epsilon > 0, "epsilon must have a positive value."

    g = lambda x: x - (f(x) / central_difference(f, x, h))
    return fixed_point_iteration(g, x_0, epsilon, N_max)