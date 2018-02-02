"""Test errors."""
from pybotics.pybotics_error import PyboticsError

from pytest import raises


def test_pybotics_error():
    """
    Test error.

    :return:
    """
    with raises(PyboticsError):
        raise PyboticsError()

    assert str(PyboticsError()) is PyboticsError._default_message
    assert str(PyboticsError('test')) is 'test'
