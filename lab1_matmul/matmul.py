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


if __name__ == "__main__":
    for size in [10, 100]:
        print(size)
        times = {key: [] for key in ["ijk", "ikj", "jik", "jki", "kij", "kji"]}
        for order in ["ijk", "ikj", "jik", "jki", "kij", "kji"]:
            for _ in range(10):
                A = np.random.rand(size, size)
                B = np.random.rand(size, size)

                start = process_time()
                C = matmul_3_loops(A, B, order)
                end = process_time()

                ms = (end - start) * 1000
                times[order].append(ms)
        for order in ["ijk", "ikj", "jik", "jki", "kij", "kji"]:
            print("\t", order, mean(times[order]), "ms")
