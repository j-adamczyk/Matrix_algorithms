import numpy as np
from scipy.linalg import lu

from lab2_Gauss_elim_and_LU_decomp.pivoting import pivoting


def gauss_elimination_classic(A: np.ndarray) \
        -> np.ndarray:
    """
    Performs Gaussian elimination on matrix A in the classic version
    with pivoting.

    :param A: square matrix of shape (n, n)
    :return: matrix A after Gaussian elimination
    """
    A = A.copy()
    n = A.shape[0]
    for k in range(n):
        A = pivoting(A, k, "row")
        Akk = A[k, k]
        for j in range(k + 1, n):
            A[j, k:] -= A[k, k:] * A[j, k] / Akk

    return A


def gauss_elimination_row(A: np.ndarray) \
        -> np.ndarray:
    """
    Performs Gaussian elimination on matrix A row-wise with pivoting.

    :param A: square matrix of shape (n, n)
    :return: matrix A after Gaussian elimination
    """
    A = A.copy()
    n = A.shape[0]
    for k in range(n):
        A = pivoting(A, k, "row")
        A[k, k:] = A[k, k:] / A[k, k]
        for j in range(k + 1, n):
            A[j, k:] -= A[k, k:] * A[j, k]
    return A


def gauss_elimination_column(A: np.ndarray) \
        -> np.ndarray:
    """
    Performs Gaussian elimination on matrix A column-wise with pivoting.

    :param A: square matrix of shape (n, n)
    :return: matrix A after Gaussian elimination
    """
    A = A.copy()
    n = A.shape[0]
    for k in range(n):
        A = pivoting(A, k, "col")
        A[k + 1:n, k] = A[k + 1:n, k] / A[k, k]
        for j in range(k + 1, n):
            A[k + 1:n, j] -= A[k + 1:n, k] * A[k, j]

    return A


A = np.array([[1, 1, 1],
              [2, 3, 7],
              [1, 2, 3]], dtype=np.float32)

A_custom = gauss_elimination_column(A.copy())

