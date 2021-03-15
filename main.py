from typing import Tuple

import numpy as np


def gaussian_elimination_gen_U_1(
        A: np.ndarray,
        b: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs Gaussian elimination on matrices A and b row-wise.
    This version creates 1 on the diagonal of the U matrix generated
    by the elimination.

    :param A: square matrix of shape (n, n)
    :param b: vector of shape (n,)
    :return: matrix A after Gaussian elimination
    """
    A = A.astype(np.float)
    b = b.astype(np.float)
    n = A.shape[0]

    for k in range(n):
        Akk = A[k, k]
        A[k, k:] /= Akk
        b[k] /= Akk
        for j in range(k + 1, n):
            Ajk = A[j, k]
            A[j] -= A[k] * Ajk
            b[j] -= b[k] * Ajk

    return A, b


def gaussian_elimination_gen_U_det(
        A: np.ndarray,
        b: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs Gaussian elimination on matrices A and b row-wise.
    This version creates elements on the diagonal of the U matrix that
    equal the determinant when multiplied.

    :param A: square matrix of shape (n, n)
    :param b: vector of shape (n,)
    :return: matrix A after Gaussian elimination
    """
    A = A.astype(np.float)
    b = b.astype(np.float)
    n = A.shape[0]

    for k in range(n):
        Akk = A[k, k]
        for j in range(k + 1, n):
            multiplier = A[j, k] / Akk
            A[j] -= A[k] * multiplier
            b[j] -= b[j] * multiplier

    return A, b


def gaussian_elimination_with_pivoting(
        A: np.ndarray,
        b: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs Gaussian elimination on matrices A and b row-wise.
    This version creates elements on the diagonal of the U matrix that
    equal the determinant when multiplied.

    :param A: square matrix of shape (n, n)
    :param b: vector of shape (n,)
    :return: matrix A after Gaussian elimination
    """
    A = A.astype(np.float)
    b = b.astype(np.float)
    n = A.shape[0]

    for k in range(n):
        A, b = pivoting_Ab(A, b, k)
        Akk = A[k, k]
        A[k, k:] /= Akk
        b[k] /= Akk
        for j in range(k + 1, n):
            Ajk = A[j, k]
            A[j] -= A[k] * Ajk
            b[j] -= b[k] * Ajk

    return A, b


def pivoting_Ab(A: np.ndarray,
               b: np.ndarray,
               k: int) -> np.ndarray:
    """
    Performs pivoting on matrix A for k-th step. For row version pivot comes
    from k-th column (from k-th row downwards), for column version it comes
    from k-th row (from k-th column rightwards).

    :param A: matrix to find pivot for
    :param b: right-hand vector
    :param k: algorithm step, row / column for pivoting
    :return: matrix A after pivoting, i.e. exchanging rows for optimal
    (largest) pivot
    """
    A = A.copy()
    b = b.copy()
    n = A.shape[0]

    max_i = k
    for i in range(k, n):
        if abs(A[i, k]) > abs(A[max_i, k]):
            max_i = i

    if max_i != k:
        A[[k, max_i], k:] = A[[max_i, k], k:]
        b[[k, max_i]] = b[[max_i, k]]

    return A, b


def LU_decomposition(A: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs LU decomposition on matrix on A with pivoting, saving results in
    a new matrix.

    :param A: square matrix of shape (n, n)
    :return: matrix with L and U matrices
    """
    A = A.astype(np.float)
    n = A.shape[0]
    L = np.zeros(A.shape)
    U = np.zeros(A.shape)
    for k in range(n):
        L[k, k] = 1
        L[k + 1:, k] = A[k + 1:, k] / A[k, k]
        U[k, k:] = A[k, k:]
        A[k + 1:, k] = 0
        for j in range(k + 1, n):
            A[k + 1:, j] -= L[k + 1:, k] * U[k, j]
    return L, U


if __name__ == '__main__':
    # Gaussian elimination, generating 1 on diagonal of U
    A = np.array([[3, 2, 1],
                  [2, 3, 1],
                  [1, 2, 3]])
    b = np.array([39, 34, 26])

    A, b = gaussian_elimination_gen_U_1(A, b)
    print("Gaussian elimination, generating 1 on diagonal")
    print("Generated:")
    print(A, "\n")
    print("Answer:")
    print(np.array([[1, 2/3, 1/3],
                    [0, 1, 0.2],
                    [0, 0, 1]]))

    print("\n\n")

    # Gaussian elimination, generating determinant on diagonal of U
    A = np.array([[2, -2, 1],
                  [0, 4, 1],
                  [1, 1, 3]])
    b = np.array([1, 1, 1])  # not used in the answer

    A, b = gaussian_elimination_gen_U_det(A, b)
    print("Gaussian elimination, generating determinant on diagonal")
    print("Generated:")
    print(A, "\n")
    print("Answer:")
    print(np.array([[2, -2, 1],
                    [0, 4, 1],
                    [0, 0, 2]]))

    print("\n\n")

    # Gaussian elimination with pivoting

    A = np.array([[6, 2, 2],
                  [6, 2, 1],
                  [1, 2, -1]])
    b = np.array([0, 5, 0])

    A, b = gaussian_elimination_with_pivoting(A, b)
    print("Gaussian elimination with pivoting")
    print("Generated:")
    print(A, "\n", b, "\n")
    print("Answer:")
    print(np.array([[1, 1/3, 1/3],
                    [0, 1, -4/5],
                    [0, 0, 1]]))
    print(np.array([0, 0, -5]))

    print("\n\n")

    # LU decomposition without pivoting

    A = np.array([[4, 3],
                  [6, 3]])

    L, U = LU_decomposition(A)

    print("LU decomposition")
    print("Generated:")
    print(L, "\n", U, "\n")
    print("Answer:")
    print(np.array([[1, 0],
                    [1.5, 1]]))
    print(np.array([[4, 3],
                    [0, -1.5]]))

    print("\n\n")
