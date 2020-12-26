from abc import ABC, abstractmethod
import numpy as np


class SparseMatrix(ABC):
    @abstractmethod
    def __init__(self):
        self.dtype = None
        self.shape = None

        self.rows = []
        self.cols = []
        self.vals = []
        self.non_zeros = None

    @abstractmethod
    def to_dense(self):
        raise NotImplementedError()


class CoordinateSparseMatrix(SparseMatrix):
    def __init__(self, matrix=None):
        """
        Constructs sparse matrix in the coordinate format from a dense matrix.
        :param matrix: 2D Numpy array
        """
        super().__init__()
        if matrix is not None:
            self.dtype = matrix.dtype
            self._from_dense(matrix)

    def _from_dense(self, matrix):
        self.shape = matrix.shape

        self.rows = []
        self.cols = []
        self.vals = []

        self.non_zeros = 0

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if not np.isclose(matrix[i, j], 0):
                    self.rows.append(i)
                    self.cols.append(j)
                    self.vals.append(matrix[i, j])
                    self.non_zeros += 1

    def to_dense(self):
        dense = np.zeros(self.shape, dtype=self.dtype)
        for i in range(self.non_zeros):
            row = self.rows[i]
            col = self.cols[i]
            val = self.vals[i]
            dense[row, col] = val

        return dense

    def get(self,
            row: int,
            col: int,
            index: bool = False):
        for i in range(len(self.vals)):
            if self.rows[i] == row and self.cols[i] == col:
                if index:
                    return self.vals[i], i
                else:
                    return self.vals[i]

        if index:
            return 0, -1
        else:
            return 0

    def insert(self,
               row: int,
               col: int,
               val: float):
        i = 0
        while True:
            while i < self.non_zeros and self.rows[i] < row:
                i += 1

            if i == self.non_zeros:
                insert_index = self.non_zeros
                break
            elif self.rows[i] > row:
                insert_index = i
                break

            while i < self.non_zeros and self.cols[i] < col:
                i += 1

            if i == self.non_zeros:
                insert_index = self.non_zeros
                break
            elif self.cols[i] > col:
                insert_index = i
                break

        self.rows.insert(insert_index, row)
        self.cols.insert(insert_index, col)
        self.vals.insert(insert_index, val)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        string = "NNZ = " + str(self.non_zeros) + "\n"
        string += "Rows: " + str(self.rows) + "\n"
        string += "Cols: " + str(self.cols) + "\n"
        string += "Values: " + str(self.vals)
        return string


class CSRMatrix(SparseMatrix):
    def __init__(self, matrix=None):
        """
        Constructs sparse matrix in the Compressed Sparse Row (CSR) format
        from a dense matrix.
        :param matrix: 2D Numpy array
        """
        super().__init__()
        if matrix is not None:
            self.dtype = matrix.dtype
            self._from_dense(matrix)

    def _from_dense(self, matrix):
        self.shape = matrix.shape

        self.rows = [0]
        self.cols = []
        self.vals = []

        self.non_zeros = 0

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if not np.isclose(matrix[i, j], 0):
                    self.vals.append(matrix[i, j])
                    self.cols.append(j)
                    self.non_zeros += 1
            self.rows.append(self.non_zeros)

    def to_dense(self):
        dense = np.zeros(self.shape, dtype=self.dtype)
        for i in range(self.shape[0]):
            row_start = self.rows[i]
            row_end = self.rows[i + 1]
            cols = self.cols[row_start:row_end]
            vals = self.vals[row_start:row_end]
            for col, val in zip(cols, vals):
                dense[i, col] = val
        return dense

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        string = "NNZ = " + str(self.non_zeros) + "\n"
        string += "Rows: " + str(self.rows) + "\n"
        string += "Cols: " + str(self.cols) + "\n"
        string += "Values: " + str(self.vals)
        return string


class CSCMatrix(SparseMatrix):
    def __init__(self, matrix=None):
        """
        Constructs sparse matrix in the Compressed Sparse Column (CSC) format
        from a dense matrix.
        :param matrix: 2D Numpy array
        """
        super().__init__()
        if matrix is not None:
            self.dtype = matrix.dtype
            self._from_dense(matrix)

    def _from_dense(self, matrix):
        self.shape = matrix.shape

        self.rows = []
        self.cols = [0]
        self.vals = []

        self.non_zeros = 0

        for j in range(matrix.shape[1]):
            for i in range(matrix.shape[0]):
                if not np.isclose(matrix[i, j], 0):
                    self.vals.append(matrix[i, j])
                    self.rows.append(i)
                    self.non_zeros += 1
            self.cols.append(self.non_zeros)

    def to_dense(self):
        dense = np.zeros(self.shape, dtype=self.dtype)
        for j in range(self.shape[1]):
            col_start = self.cols[j]
            col_end = self.cols[j + 1]
            rows = self.rows[col_start:col_end]
            vals = self.vals[col_start:col_end]
            for row, val in zip(rows, vals):
                dense[row, j] = val
        return dense

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        string = "NNZ = " + str(self.non_zeros) + "\n"
        string += "Rows: " + str(self.rows) + "\n"
        string += "Cols: " + str(self.cols) + "\n"
        string += "Values: " + str(self.vals)
        return string


def coordinate_to_CSC(coord_matrix: CoordinateSparseMatrix):
    """
    Converts sparse matrix in the coordinate format to the CSC (Compressed
    Sparse Column) format.

    :param coord_matrix: sparse matrix in the coordinate format
    :return: sparse matrix in the CSC format
    """
    csc_matrix = CSCMatrix()

    csc_matrix.dtype = coord_matrix.dtype
    csc_matrix.shape = coord_matrix.shape

    csc_matrix.rows = coord_matrix.rows
    csc_matrix.vals = coord_matrix.vals
    csc_matrix.non_zeros = coord_matrix.non_zeros

    curr_col_sum = 0
    csc_matrix.cols = []
    for col in coord_matrix.cols:
        curr_col_sum += col
        csc_matrix.cols.append(curr_col_sum)
    csc_matrix.cols.append(csc_matrix.non_zeros)

    return csc_matrix
