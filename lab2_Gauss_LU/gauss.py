import numpy as np

from lab2_Gauss_LU.pivoting import pivoting


def gauss_elimination_classic(A: np.ndarray) \
        -> np.ndarray:
    """
    Performs Gaussian elimination on matrix A in the classic version
    with pivoting.

    :param A: square matrix of shape (n, n)
    :return: matrix A after Gaussian elimination
    """
    A = A.copy().astype(np.float)
    n = A.shape[0]
    for k in range(n):
        A = pivoting(A, k, "row")
        Akk = A[k, k]
        for j in range(k + 1, n):
            A[j, k:] -= A[k, k:] * A[j, k] / Akk

    return A


def gauss_elimination_row(A: np.ndarray, use_pivoting: bool = True) \
        -> np.ndarray:
    """
    Performs Gaussian elimination on matrix A row-wise, optionally with
    pivoting.

    :param A: square matrix of shape (n, n)
    :param use_pivoting: whether to use pivoting or not
    :return: matrix A after Gaussian elimination
    """
    A = A.copy().astype(np.float)
    n = A.shape[0]
    for k in range(n - 1):
        if use_pivoting:
            A = pivoting(A, k, "row")
        Akk = A[k, k]
        A[k, k:] = A[k, k:] / Akk
        for j in range(k + 1, n):
            A[j, k + 1:] -= A[k, k + 1:] * A[j, k]
    return A


def gauss_elimination_column(A: np.ndarray) \
        -> np.ndarray:
    """
    Performs Gaussian elimination on matrix A column-wise with pivoting.

    :param A: square matrix of shape (n, n)
    :return: matrix A after Gaussian elimination
    """
    A = A.copy().astype(np.float)
    n = A.shape[0]
    for k in range(n):
        A = pivoting(A, k, "col")
        A[k + 1:n, k] = A[k + 1:n, k] / A[k, k]
        for j in range(k + 1, n):
            A[k + 1:n, j] -= A[k + 1:n, k] * A[k, j]

    return A
