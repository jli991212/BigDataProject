
from itertools import product, chain
from copy import deepcopy


class ALS_Matrix(object):
    def __init__(self, inputdata):
        self.inputdata = inputdata
        self.shape = (len(inputdata), len(inputdata[0]))

    def row(self, row_no):
        return ALS_Matrix([self.inputdata[row_no]])

    def column(self, column_no):

        m = self.shape[0]
        return ALS_Matrix([[self.inputdata[i][column_no]] for i in range(m)])

    @property
    def is_square(self):
        return self.shape[0] == self.shape[1]

    @property
    def transpose(self):
        inputdata = list(map(list, zip(*self.inputdata)))
        return ALS_Matrix(inputdata)

    def _eye(self, n):
       return [[0 if a != b else 1 for b in range(n)] for a in range(n)]

    @property
    def eye(self):
        assert self.is_square, "matrix need to square "
        inputdata = self._eye(self.shape[0])
        return ALS_Matrix(inputdata)

    def _gaussian_elimination(self, my_matrix):

        n = len(my_matrix)
        m = len(my_matrix[0])

        for column_idx in range(n):
            if my_matrix[column_idx][column_idx] == 0:
                row_idx = column_idx
                while row_idx < n and my_matrix[row_idx][column_idx] == 0:
                    row_idx += 1
                for i in range(column_idx, m):
                    my_matrix[column_idx][i] += my_matrix[row_idx][i]

            for i in range(column_idx + 1, n):
                if my_matrix[i][column_idx] == 0:
                    continue
                k = my_matrix[i][column_idx] / my_matrix[column_idx][column_idx]
                for j in range(column_idx, m):
                    my_matrix[i][j] -= k * my_matrix[column_idx][j]

        for column_idx in range(n - 1, -1, -1):
            for i in range(column_idx):
                if my_matrix[i][column_idx] == 0:
                    continue
                k = my_matrix[i][column_idx] / my_matrix[column_idx][column_idx]
                for j in chain(range(i, column_idx + 1), range(n, m)):
                    my_matrix[i][j] -= k * my_matrix[column_idx][j]

        for i in range(n):
            k = 1 / my_matrix[i][i]
            my_matrix[i][i] *= k
            for j in range(n, m):
                my_matrix[i][j] *= k

        return my_matrix

    def _inverse(self, inputdata):
        n = len(inputdata)
        unit_matrix = self._eye(n)
        my_matrix = [x + y for x, y in zip(self.inputdata, unit_matrix)]
        ret = self._gaussian_elimination(my_matrix)
        return list(map(lambda x: x[n:], ret))

    @property
    def inverse(self):

        assert self.is_square, "matrix is square now!"
        inputdata = self._inverse(self.inputdata)
        return ALS_Matrix(inputdata)

    def _row_mul(self, row1, row2):

        return sum(x[0] * x[1] for x in zip(row1, row2))

    def _matrix_mul(self, row, mat):
        row_pairs = product([row], mat.transpose.inputdata)
        return [self._row_mul(*row_pair) for row_pair in row_pairs]

    def matrix_mul(self, mat):

        error_msg = "two matrix coulmns number not match"
        assert self.shape[1] == mat.shape[0], error_msg
        return ALS_Matrix([self._matrix_mul(row, mat) for row in self.inputdata])



