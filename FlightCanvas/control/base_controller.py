from abc import ABC, abstractmethod

import numpy as np

from FlightCanvas.vehicle.vehicle_dynamics import VehicleDynamics
from typing import Union


class BaseController(ABC):
    dt: float

    def __init__(self, vehicle_dynamics: VehicleDynamics):
        self.vehicle_dynamics = vehicle_dynamics

    @abstractmethod
    def get_control(self, state: np.ndarray) -> Union[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def init_controller(self, dt: float):
        self.dt = dt
