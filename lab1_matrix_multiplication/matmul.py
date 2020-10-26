import numpy as np


def dot(vec1: np.ndarray, vec2: np.ndarray) \
        -> float:
    """
    Calculates the dot product between vectors.

    :param vec1: vector of shape (n,)
    :param vec2: vector of shape (n,)
    :return: dot product of vectors as a single float number
    """
    dot_product = 0
    for i in range(vec1.shape[0]):
        dot_product += vec1[i] * vec2[i]
    return dot_product


def matmul_1a(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    IJP matrix multiplication, using dot product between rows of A and
    columns of B.
    :param A: matrix of shape (m, k)
    :param B: matrix of shape (k, n)
    :return: matrix of shape (m, n)
    """
    m = A.shape[0]
    k = A.shape[1]  # B.shape[0]
    n = B.shape[1]

    C = np.zeros((m, n))

    for i in range(m):
        for j in range(n):
            # slice 0:k is mathematical range [0, k-1]
            C[i, j] += dot(A[i, 0:k], B[0:k, j].flatten())

    return C


A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

C = matmul_1a(A, B)

target = np.array([[19, 22],
                   [43, 50]])

assert (C == target).all()

