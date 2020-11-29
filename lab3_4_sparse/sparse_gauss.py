from copy import deepcopy
from typing import Union

import numpy as np

from .sparse_matrices import CoordinateSparseMatrix, CSRMatrix


def sparse_gauss_elimination_row(A: Union[CoordinateSparseMatrix, CSRMatrix]) \
        -> Union[CoordinateSparseMatrix, CSRMatrix]:
    """
    Performs Gaussian elimination on sparse matrix A row-wise. Allows either
    coordinate format or CSR format.

    :param A: sparse square matrix of shape (n, n)
    :return: matrix A after Gaussian elimination
    """
    A = deepcopy(A)
    A.dtype = np.float
    if isinstance(A, CoordinateSparseMatrix):
        A = _coordinate_row(A)
    elif isinstance(A, CSRMatrix):
        pass  # not yet implemented

    return A


def _coordinate_row(A: CoordinateSparseMatrix) -> CoordinateSparseMatrix:
    """
    Performs Gaussian elimination on sparse matrix A in the coordinate format
    row-wise.

    :param A: sparse square matrix in the coordinate format of shape (n, n)
    :return: matrix A after Gaussian elimination
    """
    n = A.shape[0]
    for k in range(n - 1):
        Akk = A.get(k, k)
        assert Akk != 0, "Akk = 0"

        for i in range(k + 1, n):
            Aki, ki_index = A.get(k, i, index=True)
            if ki_index != -1:
                A.vals[ki_index] /= Akk

        for j in range(k + 1, n):
            Ajk = A.get(j, k)
            if Ajk == 0:
                continue

            for i in range(k + 1, n):
                Aji, ji_index = A.get(j, i, index=True)
                if Aji != 0:
                    Aki = A.get(k, i)
                    if Aki != 0:
                        A.vals[ji_index] -= Aki * Ajk

    return A

