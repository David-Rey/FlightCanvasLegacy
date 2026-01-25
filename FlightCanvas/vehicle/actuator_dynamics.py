import numpy as np
from typing import Union


class ActuatorDynamics:
    """
    Manager class for a collection of vehicle actuators
    """
    def __init__(self, actuators: list["SISOSystem"]):
        """
        Initialize the ActuatorDynamics object given list of actuators
        :param actuators: list of actuators
        """
        self.actuators = actuators
        self.num_actuators = len(actuators)

    def c2d(self, dt: float):
        """
        Discretizes all managed actuators from continuous to discrete time
        :param dt: The sample time in seconds

        """
        for i in range(self.num_actuators):
            if self.actuators[i] is not None:
                self.actuators[i] = self.actuators[i].c2d(dt)

    def update_deflections(self, deflections: Union[np.ndarray, list]) -> np.ndarray:
        """
        Propagates the commanded deflections through the actuator dynamics
        """
        true_deflections = np.zeros(self.num_actuators)
        for i in range(self.num_actuators):
            if self.actuators[i] is not None:
                true_deflections[i] = self.actuators[i].update(deflections[i])
            else:
                true_deflections[i] = deflections[i]
        return true_deflections

    def get_true_deflections(self) -> np.ndarray:
        """
        Retrieves the most recent output (y_k) from each actuator's history
        """
        true_deflections = np.zeros(self.num_actuators)
        for i in range(self.num_actuators):
            if self.actuators[i] is not None:
                true_deflections[i] = self.actuators[i].y_hist[0]
        return true_deflections
