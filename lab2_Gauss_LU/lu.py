import numpy as np

from lab2_Gauss_LU.pivoting import pivoting


def LU_decomposition(A: np.ndarray) \
        -> np.ndarray:
    """
    Performs LU decomposition on matrix on A with pivoting, saving results in
    a new matrix.

    :param A: square matrix of shape (n, n)
    :return: matrix with L and U matrices
    """
    A = A.copy()
    n = A.shape[0]
    L = np.zeros(A.shape)
    U = np.zeros(A.shape)
    for k in range(n):
        A = pivoting(A, k, version="row")
        L[k, k] = 1
        L[k + 1:, k] = A[k + 1:, k] / A[k, k]
        U[k, k:] = A[k, k:]
        A[k + 1:, k] = 0
        for j in range(k + 1, n):
            A[k + 1:, j] -= L[k + 1:, k] * U[k, j]
    return A
