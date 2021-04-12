from typing import Tuple

import numpy as np
from numpy.linalg import norm, eig


def eigenvalue(A, v):
    return np.dot(v, np.dot(A, v)) / np.dot(v, v)


def eigendecomp(A: np.ndarray,
                eps: float = 0.01) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the eigendecomposition of matrix A using power method.

    :param A: matrix of shape (n, n)
    :param eps: precision of iterations
    :return: tuple vector, matrix - (eigenvalues, eigenvectors)
    """
    n = A.shape[0]

    eigvals = np.zeros(n)
    eigvecs = np.zeros(A.shape)

    for i in range(n):
        eig_vec = np.random.rand(n)
        eig_val = eigenvalue(A, eig_vec)

        while True:
            Av = A.dot(eig_vec)

            eig_vec_new = Av / np.linalg.norm(Av)
            eig_val_new = eigenvalue(A, eig_vec_new)

            if np.abs(eig_val - eig_val_new) < eps:
                break

            eig_vec = eig_vec_new
            eig_val = eig_val_new

        eigvals[i] = eig_val_new
        eigvecs[i] = eig_vec_new

        A = A - eig_val_new * eig_vec_new * eig_vec_new[:, np.newaxis]

    return eigvals, eigvecs


if __name__ == '__main__':
    A = np.array([[2, -1, 0],
                  [-1, 2, -1],
                  [0, -1, 2]])

    eig_vals, eig_vecs = eig(A)
    print("Library:")
    print(eig_vals)
    print(eig_vecs)
    print()

    eig_vals, eig_vecs = eigendecomp(A)
    print("My:")
    print(eig_vals)
    print(eig_vecs)
    print()
