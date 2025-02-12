# Root Finding

## Functions

### `binary_search(f: Callable, a: float, b: float, epsilon: float) -> float`
Finds an approximation for a root of the function `f` within the interval [a, b] using the bisection method.

#### Arguments:
- `f`: The function whose root is to be found.
- `a`, `b`: The endpoints of the interval. `f(a)` and `f(b)` must have opposite signs.
- `epsilon`: The tolerance for stopping the search.

#### Returns:
- The approximate root of the function `f`.

---

### `central_difference(f: Callable, x: float, epsilon: float) -> float`
Finds an approximation for the derivative of the function `f` at `x` using the central difference method.

#### Arguments:
- `f`: The function to differentiate.
- `x`: The point at which to compute the derivative.
- `epsilon`: The step size for the central difference.

#### Returns:
- The approximate derivative of `f` at `x`.

---

### `fixed_point_iteration(f: Callable, x_0: float, epsilon: float, N_max: int) -> float`
Finds an approximation for a fixed point of the function `f` using fixed-point iteration.

#### Arguments:
- `f`: The function for which a fixed point is to be found.
- `x_0`: The initial estimate.
- `epsilon`: The tolerance for stopping the iteration.
- `N_max`: The maximum number of iterations.

#### Returns:
- The approximate fixed point of the function `f`.

---

### `newton_raphson(f: Callable, x_0: float, epsilon: float, N_max: int, h: float = 1e-05) -> float`
Finds an approximation of the root of the function `f` using the Newton-Raphson method.

#### Arguments:
- `f`: The function whose root is to be found.
- `x_0`: The initial guess.
- `epsilon`: The tolerance for stopping the iteration.
- `N_max`: The maximum number of iterations.
- `h`: The step size for numerical differentiation (default: `1e-5`).

#### Returns:
- The approximate root of the function `f`.

---

# Differential Equations

## Functions

### `adams_bashforth(f: Callable, x_0: float, y_0: npt.ArrayLike, h: float, n: int) -> Dict[float, float | numpy.ndarray]`
Approximates the solution to a differential equation using the Adams-Bashforth method.

#### Arguments:
- `f`: The derivative function.
- `x_0`: The initial x-value.
- `y_0`: The initial y-value(s).
- `h`: The step size.
- `n`: The number of steps.

#### Returns:
- A dictionary mapping x-values to their corresponding y-values.

---

### `euler_method(f: Callable, x_0: float, y_0: npt.ArrayLike, h: float, n: int) -> Dict[float, float | numpy.ndarray]`
Approximates the solution to a differential equation using Euler's method.

#### Arguments:
- `f`: The derivative function.
- `x_0`: The initial x-value.
- `y_0`: The initial y-value(s).
- `h`: The step size.
- `n`: The number of steps.

#### Returns:
- A dictionary mapping x-values to their corresponding y-values.

---

### `runge_kutta(f: Callable, x_0: float, y_0: npt.ArrayLike, h: float, n: int) -> Dict[float, float | numpy.ndarray]`
Approximates the solution to a differential equation using the Runge-Kutta method.

#### Arguments:
- `f`: The derivative function.
- `x_0`: The initial x-value.
- `y_0`: The initial y-value(s).
- `h`: The step size.
- `n`: The number of steps.

#### Returns:
- A dictionary mapping x-values to their corresponding y-values.

---

# Linear Algebra

## Functions

### `gaussian_elimination(A: list[list[float]]) -> list[list[float]]`
Performs Gaussian elimination on the matrix `A` to reduce it to row echelon form.

#### Arguments:
- `A`: A 2D list representing the matrix.

#### Returns:
- The reduced row echelon form of `A`.

---

### `kernel_basis(A: list[list[float]]) -> numpy.ndarray`
Computes a basis for the kernel (null space) of the matrix `A`.

#### Arguments:
- `A`: A 2D list representing the matrix.

#### Returns:
- A basis for the kernel of `A` as a NumPy array.