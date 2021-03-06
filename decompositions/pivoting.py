import numpy as np


def pivoting_A(A: np.ndarray,
               k: int,
               version: str = "row") -> np.ndarray:
    """
    Performs pivoting on matrix A for k-th step. For row version pivot comes
    from k-th column (from k-th row downwards), for column version it comes
    from k-th row (from k-th column rightwards).

    :param A: matrix to find pivot for
    :param k: algorithm step, row / column for pivoting
    :param version: "row" if algorithm needing pivoting works on rows (find
    pivot in column)  or "col" if algorithm works on columns (find pivot in
    row)
    :return: matrix A after pivoting, i.e. exchanging rows for optimal
    (largest) pivot
    """
    A = A.copy()
    n = A.shape[0]
    if version == "row":
        max_i = k
        for i in range(k, n):
            if abs(A[i, k]) > abs(A[max_i, k]):
                max_i = i

        if max_i != k:
            A[[k, max_i], k:] = A[[max_i, k], k:]

    elif version == "col":
        max_i = k
        for i in range(k, n):
            if abs(A[k, i]) > abs(A[k, max_i]):
                max_i = i

        if max_i != k:
            A[k:, [k, max_i]] = A[k:, [max_i, k]]

    return A


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
