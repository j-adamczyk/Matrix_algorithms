from typing import Union

import numpy as np
import numpy.linalg
import scipy.linalg


def norm(A: np.ndarray,
         p: Union[int, str]) -> float:
    """
    Calculates the induced p-norm of matrix A.

    :param A: matrix to calculate the norm for
    :param p: either the number for L1, L2, ..., Lp norm or "inf" string for
    the infinity norm
    :return: p-norm of matrix A
    """
    A = A.astype(np.float)

    if p == 1:
        return np.max(np.sum(np.abs(A), axis=0))
    elif p == "inf":
        return np.max(np.sum(np.abs(A), axis=1))

    xs = np.random.randn(A.shape[0], 1000)

    # compute the norms and normalized A
    norm_xs = np.sum(np.abs(xs) ** p, axis=0) ** (1 / p)
    normalized_xs = xs / norm_xs

    # apply A to normalized vectors, i.e. calculate Ax
    Ax = A.dot(normalized_xs)

    # compute norms of Ax vectors
    norm_Ax = np.sum(np.abs(Ax) ** p, axis=0) ** (1 / p)

    # get the highest norm
    p_norm = np.max(norm_Ax)

    return p_norm


if __name__ == '__main__':
    A = np.array([[1, 2],
                  [3, 4]])

    print("My:", norm(A, 1), ", library:", np.linalg.norm(A, ord=1))
    print("My:", norm(A, 2), ", library:", np.linalg.norm(A, ord=2))
    print("My:", norm(A, "inf"), ", library:", np.linalg.norm(A, ord=np.inf))
