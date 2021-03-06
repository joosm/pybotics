"""Calibration test."""
import numpy as np
from pytest import raises

from pybotics.calibration import compute_absolute_errors
from pybotics.constants import POSITION_VECTOR_LENGTH
from pybotics.errors import ShapeMismatchError, SequenceError


def test_compute_absolute_errors(serial_robot):
    """
    Test compute_absolute_errors().

    :param serial_robot:
    :return:
    """
    num_measurements = 10
    joints = np.deg2rad(list(range(serial_robot.num_dof)))
    fk_transform = serial_robot.fk(joints)

    joints = np.tile(joints, (num_measurements, 1))
    desired_positions = np.tile(fk_transform[:-1, -1], (num_measurements, 1))

    # perfect fk
    errors = compute_absolute_errors(serial_robot, joints, desired_positions)
    np.testing.assert_allclose(errors, np.zeros(num_measurements), atol=1e-6)

    # bad fk
    distance_offset = 10
    serial_robot.world_frame.position = [distance_offset, 0, 0]
    errors = compute_absolute_errors(serial_robot, joints, desired_positions)
    np.testing.assert_allclose(errors,
                               distance_offset * np.ones(num_measurements),
                               atol=1e-6)

    # bad fk
    distance_offset = [10, 10, 10]
    serial_robot.world_frame.position = distance_offset
    errors = compute_absolute_errors(serial_robot, joints, desired_positions)
    np.testing.assert_allclose(errors,
                               np.linalg.norm(distance_offset) * np.ones(
                                   num_measurements),
                               atol=1e-6)

    # shape mismatch
    with raises(ShapeMismatchError):
        compute_absolute_errors(
            serial_robot,
            np.ones((num_measurements, serial_robot.num_dof)),
            np.ones((num_measurements - 1, POSITION_VECTOR_LENGTH))
        )

    # sequence errors
    with raises(SequenceError):
        compute_absolute_errors(
            serial_robot,
            np.ones((num_measurements, serial_robot.num_dof + 1)),
            np.ones((num_measurements, POSITION_VECTOR_LENGTH))
        )
    with raises(SequenceError):
        compute_absolute_errors(
            serial_robot,
            np.ones((num_measurements, serial_robot.num_dof)),
            np.ones((num_measurements, POSITION_VECTOR_LENGTH + 1))
        )
