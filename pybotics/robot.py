"""Robot module."""
import logging
from typing import Optional, Sequence, Sized
import json

import numpy as np  # type: ignore

from pybotics.frame import Frame
from pybotics.geometry import matrix_2_euler_zyx, wrap_2_pi
from pybotics.kinematic_chain import KinematicChain
from pybotics.pybotics_error import PyboticsError
from pybotics.robot_json_encoder import RobotJSONEncoder
from pybotics.robot_optimization_mask import RobotOptimizationMask
from pybotics.tool import Tool


class Robot(Sized):
    """Robot manipulator class."""

    def __init__(self, kinematic_chain: KinematicChain,
                 tool: Optional[Tool] = None,
                 world_frame: Optional[Frame] = None) -> None:
        """
        Construct a robot instance.

        :param kinematic_chain: kinematic chain of the manipulator
        :param tool: attached tool
        :param world_frame: transform of robot base with respect to the world
        """
        self._position = np.zeros(kinematic_chain.num_dof)
        self._position_limits = np.repeat((-2 * np.pi, 2 * np.pi),
                                          kinematic_chain.num_dof
                                          ).reshape((2, -1))

        self.world_frame = Frame() if world_frame is None else world_frame
        self.kinematic_chain = kinematic_chain
        self.tool = Tool() if tool is None else tool

    def __len__(self) -> int:
        """
        Get the number of degrees of freedom.

        :return: number of degrees of freedom
        """
        return self.kinematic_chain.num_dof

    def __repr__(self) -> str:
        """
        Get the debug representation of the robot model.

        :return:
        """
        encoder = RobotJSONEncoder(sort_keys=True, indent=4)
        return encoder.encode(self)

    def __str__(self) -> str:
        """
        Get the string representation of the robot model.

        :return:
        """
        return self.__repr__()

    def apply_optimization_vector(self, vector: np.ndarray) -> None:
        """
        Update the current instance with new optimization parameters.

        :param vector: new parameters to apply
        """
        split_vector = np.split(
            vector,
            np.cumsum(
                [len(self.world_frame.optimization_vector),
                 len(self.kinematic_chain.optimization_vector)]
            ))
        self.world_frame.apply_optimization_vector(split_vector[0])
        self.kinematic_chain.apply_optimization_vector(split_vector[1])
        self.tool.apply_optimization_vector(split_vector[2])

    def fk(self, q: Optional[Sequence[float]] = None) -> np.ndarray:
        """
        Compute the forward kinematics of a given position.

        Uses the current position if None is given.
        :param q:
        :return: 4x4 transform matrix of the FK pose
        """
        # validate
        if q is None:
            q = self.position

        # gather transforms
        transforms = [self.world_frame.matrix]
        transforms.extend(self.kinematic_chain.transforms(q))
        transforms.append(self.tool.matrix)

        # matrix multiply through transforms
        pose = np.eye(4, dtype=float)
        for t in transforms:
            pose = np.dot(pose, t)

        return pose

    def ik(self, pose: np.ndarray, q: Optional[Sequence[float]] = None,
           alpha: float = 0.1, atol: float = 1e-6, max_iter: float = 1e3) -> \
            Optional[np.ndarray]:
        """Solve the inverse kinematics."""
        # set seed joints
        q = self.position if q is None else np.array(q)

        # convert pose to vector
        desired_vector = matrix_2_euler_zyx(pose)

        # solve IK
        is_running = True
        num_iterations = 0
        while is_running:
            num_iterations += 1

            # get jacobian and pseudo inverse
            j = self.jacobian_world(q)
            j_pinv = np.linalg.pinv(j)

            # get current vector
            current_pose = self.fk(q)
            current_vector = matrix_2_euler_zyx(current_pose)
            vector_diff = desired_vector - current_vector

            if np.allclose(vector_diff, 0, atol=atol):
                is_running = False
            elif num_iterations >= max_iter:
                logging.warning('Maximum number of iterations reached')
                is_running = False
            else:
                # update joints
                dq = np.dot(j_pinv, vector_diff)
                q += alpha * dq
                q = np.array([wrap_2_pi(x) for x in q])  # type: ignore

        return q

    @property
    def num_dof(self) -> int:
        """
        Get the number of degrees of freedom.

        :return: number of degrees of freedom
        """
        return self.kinematic_chain.num_dof

    @property
    def optimization_mask(self) -> RobotOptimizationMask:
        """
        Return the mask used to select the optimization parameters.

        :return: mask
        """
        mask = RobotOptimizationMask(
            world_frame=self.world_frame.optimization_mask,
            kinematic_chain=self.kinematic_chain.optimization_mask,
            tool=self.tool.optimization_mask)
        return mask

    @optimization_mask.setter
    def optimization_mask(self, value: RobotOptimizationMask) -> None:
        # FIXME: remove `# type: ignore`
        # FIXME: remove kc; it's there to shorten line length for flake8
        # https://github.com/python/mypy/issues/4167
        self.world_frame.optimization_mask = value.world_frame  # type: ignore
        kc = value.kinematic_chain
        self.kinematic_chain.optimization_mask = kc  # type: ignore
        self.tool.optimization_mask = value.tool  # type: ignore

    @property
    def optimization_vector(self) -> np.ndarray:
        """
        Return the values of parameters being optimized.

        :return: optimization parameter values
        """
        world = self.world_frame.optimization_vector
        kinematic_chain = self.kinematic_chain.optimization_vector
        tool = self.tool.optimization_vector

        vector = np.hstack((world, kinematic_chain, tool))
        return vector

    @property
    def position(self) -> np.ndarray:
        """
        Get the robot configuration (e.g., joint positions for serial robot).

        :return: robot position
        """
        return self._position

    @position.setter
    def position(self, value: np.ndarray) -> None:
        # TODO: check if position is in limits
        self._position = value

    @property
    def position_limits(self) -> np.ndarray:
        """
        Limits of the robot position (e.g., joint limits).

        :return: limits with shape (2,num_dof) where first row is upper limits
        """
        return self._position_limits

    @position_limits.setter
    def position_limits(self, value: np.ndarray) -> None:
        if value.shape[0] != 2 or value.shape[1] != len(self):
            raise PyboticsError(
                'position_limits must have shape=(2,{})'.format(len(self)))
        self._position_limits = value

    def jacobian_world(self,
                       q: Optional[Sequence[float]] = None) -> np.ndarray:
        """Calculate Jacobian with respect to the world frame."""
        # validate
        q = self.position if q is None else q

        jacobian_flange = self.jacobian_flange(q)
        pose = self.fk(q)

        rotation = pose[:3, :3]

        jacobian_transform = np.zeros((6, 6), dtype=float)

        jacobian_transform[:3, :3] = rotation
        jacobian_transform[3:, 3:] = rotation
        jacobian_world = np.dot(jacobian_transform, jacobian_flange)

        return jacobian_world

    def jacobian_flange(self,
                        q: Optional[Sequence[float]] = None) -> np.ndarray:
        """Calculate Jacobian with respect to the flange."""
        # validate
        q = self.position if q is None else q

        # init Cartesian jacobian (6-dof in space)
        jacobian_flange = np.zeros((6, self.num_dof))
        current_transform = self.tool.matrix.copy()

        for i in reversed(range(self.num_dof)):
            d = np.array([
                -current_transform[0, 0] * current_transform[1, 3] +
                current_transform[1, 0] * current_transform[0, 3],
                - current_transform[0, 1] * current_transform[1, 3] +
                current_transform[1, 1] * current_transform[0, 3],
                - current_transform[0, 2] * current_transform[1, 3] +
                current_transform[1, 2] * current_transform[0, 3],
            ])
            delta = current_transform[2, 0:3]

            jacobian_flange[:, i] = np.hstack((d, delta))

            current_link = self.kinematic_chain.links[i]
            p = q[i]  # type: ignore
            current_link_transform = current_link.transform(p)
            current_transform = np.dot(current_link_transform,
                                       current_transform)

        return jacobian_flange

    def calculate_joint_torques(
            self, q: Optional[Sequence[float]] = None,
            wrench: Optional[Sequence[float]] = None) -> np.ndarray:
        """
        Calculate the joint torques.

        Due to external force applied to the flange frame.
        Method from:
        5.9 STATIC FORCES IN MANIPULATORS
        Craig, John J. Introduction to robotics: mechanics and control.
        Vol. 3. Upper Saddle River: Pearson Prentice Hall, 2005.
        :param wrench:
        :param q:
        :return:
        """
        # validate
        q = self.position if q is None else q
        wrench = np.zeros(6) if wrench is None else wrench

        # split wrench into components
        force = wrench[:3]
        moment = wrench[-3:]

        # init output
        joint_torques = [moment[-1]]

        # loop through links from flange to base
        # each iteration calculates for link i-1
        for i, angle in reversed(list(enumerate(q))):  # pragma: no branch
            if i == 0:
                break

            # get current link transform
            transform = self.kinematic_chain.links[i].transform(angle)

            # calculate force applied to current link
            rotation = transform[:3, :3]
            force = np.dot(rotation, force)

            # calculate moment applied to current link
            linear_component = transform[:3, -1]
            moment = np.dot(rotation, moment) + np.cross(linear_component,
                                                         force)

            # append z-component as joint torque
            joint_torques.append(moment[-1])

        # reverse torques into correct order
        return np.array(list(reversed(joint_torques)), dtype=float)

    def to_json(self) -> str:
        return json.dumps(self, cls=RobotJSONEncoder, indent=4, sort_keys=True)

    def joint_range_availability(
            self,
            q: Optional[Sequence[float]] = None) -> float:
        """
        Calculate the JRA.

        Liegeois, Alain.
        "Automatic supervisory control of the configuration and behavior of
        multibody mechanisms."
        IEEE transactions on systems, man,
        and cybernetics 7.12 (1977): 868-871.
        :param q:
        :return:
        """
        # validate
        q = self.position if q is None else q

        return float(np.sum(np.square(q)))

    def joint_mid_range_proximity(
            self, q: Optional[Sequence[float]] = None,
            w: Optional[np.ndarray] = None) -> float:
        """
        Calculate the proximity to the joint mid range.

        Baron, L.
        A joint-limits avoidance strategy for arc-welding robots.
        Paper presented at the International Conference on Integrated Design
        and Manufacturing in Mechanical Engineering. Montreal
        :param q:
        :param w:
        :return:
        """
        # validate
        q = self.position if q is None else q
        w = np.eye(self.num_dof) if w is None else w

        mid_range_joints = 0.5 * np.sum(self.position_limits, axis=0)

        diff = q - mid_range_joints
        z_0 = 0.5 * diff.transpose()
        z_1 = np.dot(w, diff)
        z = np.dot(z_0, z_1)

        return float(z)

    def joint_velocity_measure(
            self, q_d: Sequence[float],
            q: Optional[Sequence[float]] = None) -> float:
        """
        Calculate the Joint Velocity Measure (JVM).

        Represents the magnitude of joint velocity
        by computing the displacement to be traveled.

        Kapoor, C., Cetin,M., Tesar, D.: Performance based redundancy
        resolution with multiple criteria. In: Design Engineering
        Technical Conference, DETC98, ASME. Georgia
        (1998)
        :param q:
        :param w:
        :return:
        """
        # validate
        q = self.position if q is None else q
        return float(np.sum(np.square(q_d - q)))

    def random_position(self, in_place: bool = False) -> Optional[np.ndarray]:
        q = np.random.uniform(
            low=self.position_limits[0],
            high=self.position_limits[1]
        )
        if in_place:
            self.position = q
        else:
            return q

    def manipulability_index(
            self, q: Optional[Sequence[float]] = None) -> float:
        """
        Calculate the Manipulability Index.

        Yoshikawa, T.: Manipulability of robotic mechanisms.
        Int. J. Robot. Res. 4(2), 3–9 (1985).
        doi:10.1177/027836498500400201
        :return:
        """
        # validate
        q = self.position if q is None else q

        jacob = self.jacobian_world(q)

        # check if jacobian is square (aka non-redundant)
        is_square_jacob = jacob.shape[0] == jacob.shape[1]
        if is_square_jacob:
            jacob_det = np.linalg.det(jacob)
            w = np.abs(jacob_det)
        else:
            jacob_2 = np.dot(jacob, jacob.transpose())
            jacob_det = np.linalg.det(jacob_2)
            w = np.sqrt(jacob_det)

        return float(w)

    def condition_number(self, q: Optional[Sequence[float]] = None) -> float:
        """
        Calculate the Condition Number.

        Salisbury, J.K., Craig, J.J.: Articulated hands: force control
        and kinematic issues. Int. J. Robot. Res. 1(1), 4–17 (1982).
        doi:10.1177/027836498200100102

        The condition number does not have an upper bound. κ ∈ [1,∞]
        :param q:
        :return:
        """
        # validate
        q = self.position if q is None else q

        jacob = self.jacobian_world(q)

        k = np.linalg.norm(jacob) * np.linalg.norm(np.linalg.pinv(jacob))

        return float(k)
