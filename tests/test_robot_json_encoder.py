"""Test robot JSON encoder."""
from pybotics.robot_json_encoder import RobotJSONEncoder

from pytest import raises


def test_default():
    """Test default encoder."""
    encoder = RobotJSONEncoder()

    with raises(TypeError):
        encoder.encode({1, 2})
