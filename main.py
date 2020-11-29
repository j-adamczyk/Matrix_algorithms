import numpy as np

from lab3_4_sparse.sparse_matrices import CoordinateSparseMatrix
from lab3_4_sparse.sparse_gauss import sparse_gauss_elimination_row

if __name__ == '__main__':
    A = np.array([[1, 3, 0, 0, 0],
                  [0, 4, 5, 0, 0],
                  [0, 0, 6, 0, 0],
                  [2, 0, 0, 7, 0],
                  [0, 0, 0, 0, 8]])

    A_coordinate = CoordinateSparseMatrix(A)
    #print(A_coordinate.to_dense())

    A_gauss = sparse_gauss_elimination_row(A_coordinate)

    print(A_gauss.to_dense())
