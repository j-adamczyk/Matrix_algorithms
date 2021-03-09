import numpy as np
from scipy.linalg import lu_factor

from sparse_matrices.sparse_matrices import CoordinateSparseMatrix
from sparse_matrices.sparse_gauss import sparse_gauss_elimination_row


if __name__ == '__main__':
    A = np.array([[1, 3, 0, 0, 0],
                  [0, 4, 5, 0, 0],
                  [0, 0, 6, 0, 0],
                  [2, 0, 0, 7, 0],
                  [0, 0, 0, 0, 8]])

    A_coordinate = CoordinateSparseMatrix(A)
    A_gauss = sparse_gauss_elimination_row(A_coordinate)
    print(A_gauss.to_dense())
