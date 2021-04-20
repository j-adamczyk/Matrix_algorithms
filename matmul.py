import numpy as np
from statistics import mean
from time import process_time


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


def matmul_3_loops(A: np.ndarray,
                   B: np.ndarray,
                   order="ijk") \
        -> np.ndarray:
    """
    IJK matrix multiplication, using dot product between rows of A and
    columns of B.
    :param A: matrix of shape (m, k)
    :param B: matrix of shape (k, n)
    :param order: order of multiplication loops
    :return: matrix of shape (m, n)
    """
    m = A.shape[0]
    l = A.shape[1]  # B.shape[0]
    n = B.shape[1]

    C = np.zeros((m, n))

    if order == "ijk":
        for i in range(m):
            for j in range(n):
                for k in range(l):
                    C[i, j] += A[i, k] * B[k, j]
    elif order == "ikj":
        for i in range(m):
            for k in range(l):
                C[i, :n] += A[i, k] * B[k, :n]
    elif order == "jik":
        for j in range(n):
            for i in range(m):
                for k in range(l):
                    C[i, j] += A[i, k] * B[k, j]
    elif order == "jki":
        for j in range(n):
            for k in range(l):
                C[:m, j] += A[:m, k] * B[k, j]
    elif order == "kij":
        for k in range(l):
            for i in range(m):
                C[i, :n] += A[i, k] * B[k, :n]
    elif order == "kji":
        for k in range(l):
            for j in range(n):
                C[:m, j] += B[:m, k] * B[k, j]

    return C


def matmul_block(A: np.ndarray,
                 B: np.ndarray) \
        -> np.ndarray:
    """
    Block matrix multiplication, using recursive definition.
    :param A: matrix of shape (n, n)
    :param B: matrix of shape (n, n)
    :return: matrix of shape (m, n)
    """
    n = A.shape[0]

    A = A.astype(np.float)
    B = B.astype(np.float)

    if n <= 2:
        return A @ B

    i = n // 2
    A_block = [[A[:i, :i], A[:i, i:]],
               [A[i:, :i], A[i:, i:]]]

    B_block = [[B[:i, :i], B[:i, i:]],
               [B[i:, :i], B[i:, i:]]]

    C = np.empty((n, n))

    C[:i, :i] = matmul_block(A_block[0][0], B_block[0][0]) + \
                matmul_block(A_block[0][1], B_block[1][0])

    C[:i, i:] = matmul_block(A_block[0][0], B_block[0][1]) + \
                matmul_block(A_block[0][1], B_block[1][1])

    C[i:, :i] = matmul_block(A_block[1][0], B_block[0][0]) + \
                matmul_block(A_block[1][1], B_block[1][0])

    C[i:, i:] = matmul_block(A_block[1][0], B_block[0][1]) + \
                matmul_block(A_block[1][1], B_block[1][1])

    return C


if __name__ == "__main__":
    A = np.arange(0, 9).reshape((3, 3))
    B = np.arange(0, 9).reshape((3, 3))

    C = A @ B
    print(C)
    print()
    print(matmul_block(A, B))
