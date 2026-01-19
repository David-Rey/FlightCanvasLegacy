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

    @abstractmethod
    def init_actor(self, pl: pv.Plotter, **kwargs):
        pass

    @abstractmethod
    def init_debug(self, pl: pv.Plotter, com: np.ndarray):
        pass

    @abstractmethod
    def update_actor(self, **kwargs):
        pass

    @abstractmethod
    def get_transform(self, **kwargs) -> Union[np.ndarray, ca.MX]:
        pass
