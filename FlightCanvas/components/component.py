from abc import ABC, abstractmethod
import numpy as np
import pyvista as pv
import casadi as ca
from typing import Union, List, Tuple
from FlightCanvas import utils


class Component(ABC):
    def __init__(
            self,
            name: str,
            xyz_ref=np.array([0., 0., 0.])
    ):
        self.name = name
        self.xyz_ref = xyz_ref
        self.parent = None

        self.static_transform_matrix = np.eye(4)
        self.dynamic_transform_matrix = np.eye(4)

    def set_parent(self, parent: 'AeroVehicle'):
        """
        Sets self.parent to the AeroVehicle for higher level information such as center of mass
        :param: parent: The AeroVehicle that the component belongs to
        """
        self.parent = parent

    def translate(self, xyz: Union[np.ndarray, List[float]]) -> "Component":
        """
        Sets the component's reference position relative to the vehicle's origin
        :param xyz: The new reference position [x, y, z] in the vehicle's body frame
        :return: The instance of the component (`self`)
        """
        self.xyz_ref = np.array(xyz)
        return self

    def set_translate(self, xyz: Union[np.ndarray, List[float]]):
        """
        Sets the component's reference position. This is an alias for `translate`
        :param xyz: The new reference position [x, y, z] in the vehicle's body frame
        """
        self.translate(xyz)

    def update_dynamic_transform(self, state: np.ndarray):
        """
        TODO move to utils
        Updates the component's dynamic transformation matrix used for animation
        :param state: The current state of the vehicle (position, velocity, quaternion, angular_velocity)
        """

        pos_I = state[:3]  # Position in the inertial frame
        quat = state[6:10]  # Orientation as a quaternion

        # Construct a transformation matrix
        R = utils.dir_cosine_np(quat)
        static_to_dynamic_transform = np.eye(4)
        static_to_dynamic_transform[:3, 3] = pos_I
        static_to_dynamic_transform[:3, :3] = R

        self.dynamic_transform_matrix = static_to_dynamic_transform @ self.static_transform_matrix

    @abstractmethod
    def update_transform(self, **kwargs):
        """
        Computes and updates the component's stored transformation matrix
        :param kwargs: Additional keyword arguments to pass to get_transform
        """
        pass

    @abstractmethod
    def init_actor(self, pl: pv.Plotter, **kwargs):
        pass

    @abstractmethod
    def init_debug(self, pl: pv.Plotter, com: np.ndarray):
        pass

    @abstractmethod
    def update_actor(self, **kwargs):
        pass

    #@abstractmethod
    #def get_transform(self, **kwargs) -> Union[np.ndarray, ca.MX]:
    #    pass
