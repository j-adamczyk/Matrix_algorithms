from typing import Tuple

import numpy as np
from numpy.linalg import svd, inv

from eigendecomposition import eigendecomp


def square_svd(A: np.ndarray) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the SVD decomposition of a square matrix using
    eigendecomposition:
    A = U * E * V^T

    :param A: square real matrix
    :return: matrices U, E, V^T; E is returned as a vector
    """
    AA_T = A @ A.T
    A_TA = A.T @ A

    E, U = eigendecomp(AA_T)
    E = np.sqrt(E)

    _, V = eigendecomp(A_TA)

    return U, E, V.T


if __name__ == '__main__':
    A = np.array([[2, -1, 0],
                  [-1, 2, -1],
                  [0, -1, 2]])

    U, E, V_T = svd(A)
    print("Library:")
    print(U)
    print(E)
    print(V_T)
    print()

    U, E, V_T = square_svd(A)
    print("My:")
    print(U)
    print(E)
    print(V_T)


