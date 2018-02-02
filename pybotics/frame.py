"""Frame module."""
from itertools import compress
from typing import Sequence, Union

import numpy as np  # type: ignore

from pybotics.geometry import euler_zyx_2_matrix, matrix_2_euler_zyx
from pybotics.orientation_convention import OrientationConvention


class Frame:
    """Frame class representing a 4x4 spatial transform."""

    def __init__(self, matrix: np.ndarray = None) -> None:
        """
        Construct a frame.

        :param matrix:
        """
        self._matrix = None
        self._optimization_mask = [False] * TRANSFORM_VECTOR_LENGTH

        self.matrix = np.eye(4) if matrix is None else matrix

    def apply_optimization_vector(self, vector: np.ndarray) -> None:
        # we are going to iterate through the given vector;
        # an iterator allows us to next()
        # (aka `pop`) the values only when desired;
        # we only update the current vector where the mask is True
        """
        Update the current instance with new optimization parameters.

        :param vector: new parameters to apply
        """
        vector_iterator = iter(vector)
        updated_vector = [v if not m else next(vector_iterator)
                          for v, m in zip(self.vector(),
                                          self.optimization_mask)]

        updated_matrix = euler_zyx_2_matrix(np.array(updated_vector))
        self.matrix = updated_matrix

    @property
    def matrix(self) -> np.ndarray:
        """
        Return the internal matrix representation of the frame.

        :return: 4x4 matrix
        """
        return self._matrix

    @matrix.setter
    def matrix(self, value: np.ndarray) -> None:
        self._matrix = value

    @property
    def optimization_mask(self) -> Sequence[bool]:
        """
        Return the mask used to select the optimization parameters.

        :return: mask
        """
        return self._optimization_mask

    @optimization_mask.setter
    def optimization_mask(self, mask: Union[bool, Sequence[bool]]) -> None:
        if isinstance(mask, bool):
            self._optimization_mask = \
                [mask] * TRANSFORM_VECTOR_LENGTH
        else:
            self._optimization_mask = list(mask)

    @property
    def optimization_vector(self) -> np.ndarray:
        """
        Return the values of parameters being optimized.

        :return: optimization parameter values
        """
        filtered_iterator = compress(self.vector(), self.optimization_mask)
        vector = np.array(list(filtered_iterator))
        return vector

    @property
    def position(self) -> np.ndarray:
        """
        Get the position XYZ of the frame.

        :return:
        """
        return self.matrix[:-1, -1]

    @position.setter
    def position(self, value: Sequence[float]) -> None:
        self.matrix[:-1, -1] = value

    def vector(self,
               convention: OrientationConvention =
               OrientationConvention.EULER_ZYX) -> np.ndarray:
        """
        Return the vector representation of the frame.

        :param convention: selectable vector convention
        :return: vectorized matrix
        """
        if convention is OrientationConvention.EULER_ZYX:
            return matrix_2_euler_zyx(self.matrix)
        else:
            raise NotImplementedError(convention)
