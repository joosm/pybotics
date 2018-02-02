"""Test robot models."""
from pybotics.robot_model import KukaLbrIiwa7, MecademicMeca500, Puma560, UR10


def test_models():
    # simply construct the models to ensure no errors
    KukaLbrIiwa7()
    UR10()
    MecademicMeca500()
    Puma560()
