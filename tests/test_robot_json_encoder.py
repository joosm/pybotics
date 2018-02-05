"""Test robot JSON encoder."""
from pybotics.robot_json_encoder import RobotJSONEncoder

from pytest import raises

import numpy as np


def test_default():
    """Test default encoder."""
    encoder = RobotJSONEncoder()

    # a set should fail
    with raises(TypeError):
        encoder.encode(set())

    # ndarrays should pass
    encoder.encode(np.array([]))

    # numpy numeric types should pass
    encoder.encode(np.int32(123))
    encoder.encode(np.float16(123))
