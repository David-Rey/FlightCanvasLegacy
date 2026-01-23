from FlightCanvas.vehicle.vehicle_dynamics import VehicleDynamics
import numpy as np


class StarshipController:
    def __init__(self, vehicle_dynamics: VehicleDynamics):
        self.vehicle_dynamics = vehicle_dynamics

    def get_control(self, state: np.ndarray) -> np.ndarray:
        pass
