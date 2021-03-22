import numpy as np
from numpy.linalg import inv


def plu(A):
    n = A.shape[0]

    P = np.eye(n, dtype=float)
    L = np.eye(n, dtype=float)
    U = A.copy()

    for i in range(n):
        # pivoting, row swapping, change P if needed
        max_el = abs(U[i, i])
        max_row = i
        for k in range(i + 1, n):
            pivot = abs(U[i, i])
            if pivot > max_el:
                max_el = pivot
                max_row = k

        if max_row != i:
            U[[max_row, i]] = U[[i, max_row]]
            P[[max_row, i]] = P[[i, max_row]]

        # forward loop, fill L and U
        factor = U[i + 1:, i] / U[i, i]
        L[i + 1:, i] = factor
        U[i + 1:] -= factor[:, np.newaxis] * U[i]

    return P, L, U


def forward_substitution(L, b):
    n = L.shape[0]

    y = np.zeros(n, dtype=np.float)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])

    return y


def back_substitution(U, y):
    n = U.shape[0]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (1. / U[i, i]) * (y[i] - np.dot(U[i, i:], x[i:]))

    return x


def inverse(A: np.ndarray) -> np.ndarray:
    """
    Computes the inverse of the A matrix, using the LU decomposition and
    solving linear equations to solve AX = B, where X is the inverse of A and
    B is an identity matrix.

    :param A: square matrix of shape (n, n)
    :return: inverse of X, it it is not singular (ValueError then)
    """
    A = A.astype(float)

    P, L, U = plu(A)
    n = A.shape[0]

    if (np.diagonal(U) == 0).any():
        raise ValueError("A is a singular value")

    A_inv = np.zeros(A.shape, dtype=float)

    for i in range(n):
        b = np.zeros(n)
        b[i] = 1

        y = forward_substitution(L, P @ b)
        x = back_substitution(U, y)

        A_inv[:, i] = x

    return A_inv


if __name__ == '__main__':
    A = np.array([[2, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

    print(inverse(A))
