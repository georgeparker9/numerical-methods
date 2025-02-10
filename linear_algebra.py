import numpy as np

def _row_swap(M: np.ndarray, i: int, j: int) -> np.ndarray:
    """
    Helper Function.
    Swaps rows i and j in the matrix M.
    """
    M[[i, j]] = M[[j, i]]


def _row_multiply(M, i: int, a: float):
    """
    Helper Function.
    Multiplies row i in the matrix M by scalar a.
    """
    M[i] *= a


def _row_subtract(M, i: int, j: int, a: float):
    """
    Helper Function.
    Subtracts a times row j from row i in the matrix M.
    """
    M[i] -= M[j] * a


def gaussian_elimination(A: list[list[float]]) -> list[list[float]]:
    """
    Performs Gaussian elimination on the matrix A to reduce it to row echelon form.

    Args:
        A: A 2D list representing the matrix.

    Returns:
        The reduced row echelon form of A.
    """
    M = np.array(A, dtype=float)
    assert M.ndim == 2, "input is not a 2D matrix."

    pivot = 0
    n, m = M.shape

    for j in range(m):
        col = M[pivot:, j]
        for i, entry in enumerate(col):
            if entry != 0:
                _row_swap(M, pivot, i + pivot)
                col = M[pivot:, j]
                _row_multiply(M, pivot, 1 / (col[pivot]))
                for k in range(pivot + 1, n):
                    _row_subtract(M, k, pivot, M[k, j])
                pivot += 1
                break
    return M


def kernel_basis(A: list[list[float]]) -> np.ndarray:
    """
    Computes a basis for the kernel (null space) of the matrix A.

    Args:
        A: A 2D list representing the matrix.

    Returns:
        A basis for the kernel of A as a NumPy array.
    """
    A_reduced = gaussian_elimination(A)
    n, m = A_reduced.shape

    pivot_cols_index = []
    pivot_rows_index = []

    for i in range(n):
        for j in range(m):
            if A_reduced[i, j] == 1:
                pivot_cols_index.append(j)
                pivot_rows_index.append(i)
                break

    param_cols_index = [j for j in range(n) if j not in pivot_cols_index]
    basis = []

    for param in param_cols_index:
        basis_member = np.zeros(m)
        basis_member[param] = 1
        for index, pivot in enumerate(reversed(pivot_cols_index)):
            basis_member[pivot] = -sum(
                [
                    A_reduced[pivot_cols_index[-(index + 1)], k] * basis_member[k]
                    for k in range(pivot + 1, m)
                ]
            )
        basis.append(basis_member)

    return np.array(basis)