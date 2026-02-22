"""
MatrixVectorizer: converts between symmetric adjacency matrices and their
upper-triangular vector representations using column-wise (vertical) traversal.

Vectorization scheme (exclude_diagonal=True, the competition default):
  - Traverse column by column, collect elements where row < col (upper triangle)
  - For a 160x160 matrix -> vector of length 160*159/2 = 12720
  - For a 268x268 matrix -> vector of length 268*267/2 = 35778

Reference: https://github.com/basiralab/DGL/blob/main/Project/MatrixVectorizer.py
"""

import numpy as np


class MatrixVectorizer:
    """Transforms symmetric matrices to/from their vector representations."""

    @staticmethod
    def vectorize(matrix: np.ndarray, include_diagonal: bool = False) -> np.ndarray:
        """
        Converts a symmetric matrix into a 1-D vector via column-wise upper-triangle
        extraction.

        Args:
            matrix:           Square symmetric matrix of shape (N, N).
            include_diagonal: If True, also includes the sub-diagonal elements
                              (row == col + 1) after each column.

        Returns:
            1-D numpy array of upper-triangular elements.
        """
        n = matrix.shape[0]
        elements = []
        for col in range(n):
            for row in range(n):
                if row < col:
                    elements.append(matrix[row, col])
                elif include_diagonal and row == col + 1:
                    elements.append(matrix[row, col])
        return np.array(elements)

    @staticmethod
    def anti_vectorize(vector: np.ndarray, matrix_size: int,
                       include_diagonal: bool = False) -> np.ndarray:
        """
        Reconstructs a symmetric matrix from its upper-triangular vector.

        Args:
            vector:           1-D array produced by vectorize().
            matrix_size:      Size N of the target (N, N) matrix.
            include_diagonal: Must match the flag used during vectorize().

        Returns:
            Symmetric numpy array of shape (matrix_size, matrix_size).
        """
        matrix = np.zeros((matrix_size, matrix_size))
        idx = 0
        for col in range(matrix_size):
            for row in range(matrix_size):
                if row < col:
                    matrix[row, col] = vector[idx]
                    matrix[col, row] = vector[idx]
                    idx += 1
                elif include_diagonal and row == col + 1:
                    matrix[row, col] = vector[idx]
                    matrix[col, row] = vector[idx]
                    idx += 1
        return matrix
