"""Pytest config."""
from typing import Tuple

import numpy as np

from pybotics.frame import Frame
from pybotics.kinematic_chain import KinematicChain
from pybotics.robot import Robot
from pybotics.robot_model import UR10
from pybotics.tool import Tool

from pytest import fixture


@fixture(name='planar_robot_link_lengths')
def planar_robot_link_lengths_fixture() -> Tuple[float, float]:
    """Get lengths for planar robot."""
    return 10, 20


@fixture(name='serial_robot')
def serial_robot_fixture():
    """
    Generate serial robot.

    :return:
    """
    return UR10()


@fixture(name='planar_robot')
def planar_robot_fixture(planar_kc):
    """
    Generate planar robot.

    :return:
    """
    return Robot(planar_kc)


@fixture(name='planar_kc')
def planar_kc_fixture(planar_robot_link_lengths):
    """
    Generate planar kinematic chain.

    :return:
    """
    mdh = np.array([
        [0, 0, 0, 0],
        [0, planar_robot_link_lengths[0], 0, 0],
        [0, planar_robot_link_lengths[1], 0, 0]
    ])
    kc = KinematicChain.from_array(mdh)
    return kc


@fixture(name='tool')
def tool_fixture():
    """
    Generate tool.

    :return:
    """
    tool = Tool()
    tool.position = [10, 20, 30]
    tool.mass = 5
    tool.cg = [-10, -20, -30]
    return tool


@fixture(name='world_frame')
def world_frame_fixture():
    """
    Generate tool.

    :return:
    """
    frame = Frame()
    frame.position = [10, 20, 30]
    return frame
