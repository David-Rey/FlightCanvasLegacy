# aero_project/FlightCanvas/aero_vehicle.py

import pathlib
from typing import List, Union, Optional

import aerosandbox.numpy as np

from FlightCanvas.components.propulsion import Propulsion
from FlightCanvas.vehicle.actuator_dynamics import ActuatorDynamics
from FlightCanvas.vehicle.vehicle_dynamics import VehicleDynamics

from FlightCanvas.components.aero_component import AeroComponent


class AeroVehicle:
    """
    Represents a complete aerodynamic vehicle composed of various FlightCanvas.
    Manages collective mesh generation and visualization.
    """

    def __init__(
            self,
            name: str,
            xyz_ref: Union[np.ndarray, List[float]],
            aero_components: List[AeroComponent],
            prop_components: Optional[List[Propulsion]] = None,
    ):
        """
        Initializes the AeroVehicle instance
        :param name: The name of the vehicle (e.g., "MyDrone")
        :param xyz_ref: The reference point [x, y, z] for the vehicle, typically the CG
        :param aero_components: A list of AeroComponent instances that comprise the vehicle
        """
        self.name = name
        self.xyz_ref = np.array(xyz_ref)
        self.mass = 10.0
        self.moi = self.mass * np.eye(3)

        self.aero_components = aero_components
        self.num_aero_components = len(aero_components)
        self.prop_components = prop_components
        self.num_prop_components = len(prop_components) if prop_components is not None else 0

        self.components = self.aero_components
        if self.num_prop_components != 0:
            self.components = self.aero_components + self.prop_components

        self.num_components = self.num_aero_components + self.num_prop_components

        self.vehicle_dynamics = None
        self.actuator_dynamics = None
        self.vehicle_path = f'vehicle_saves/{self.name}'

        [comp.set_parent(self) for comp in self.components]

        path_object = pathlib.Path(self.vehicle_path)
        path_object.mkdir(parents=True, exist_ok=True)

        # Crates a buildup manager for each component
        self.init_buildup_manager()

        # Update transformation matrices for all components
        self.update_transform()

        # Init Actuator Dynamics
        self.init_actuator_dynamics()

    def update_transform(self):
        """
        Update transformation matrices for all components
        TODO: update this to all components
        """
        [comp.update_transform() for comp in self.aero_components]

    def set_mass(self, mass: float):
        """
        Sets the mass of the vehicle
        """
        self.mass = mass

    def set_moi_factor(self, moi_factor: float):
        """
        Sets the mass moment of inertia
        """
        self.moi = moi_factor * self.mass * np.eye(3)

    def set_moi_diag(self, moi_diag: Union[np.ndarray, List[float]]):
        """
        Sets the mass moment of inertia
        """
        self.moi = np.diag(np.array(moi_diag)) * self.mass

    def init_vehicle_dynamics(self, control_mapping: Union[None, dict]):
        """
        Creates dynamics in the form x_dot = f(x, u)
        """
        self.vehicle_dynamics = VehicleDynamics(self.mass, self.moi, self.aero_components, control_mapping, self.prop_components)

    def init_buildup_manager(self):
        """
        Crates a buildup manager for each component
        """
        for aero_component in self.aero_components:
            aero_component.init_buildup_manager(self.vehicle_path, aero_component)

    def compute_buildup(self):
        """
        Computes the aerodynamic buildup data for all 'prime' aero components
        """
        print("Computing buildup data...")
        for aero_component in self.aero_components:
            aero_component.compute_buildup()

    def save_buildup(self):
        """
        Saves the aerodynamic buildup data for all 'prime' aero components
        """
        print("Saving buildup data...")
        for aero_component in self.aero_components:
            aero_component.save_buildup()

    def save_buildup_fig(self):
        """
        Saves the aerodynamic buildup figures for all 'prime' aero components
        """
        print("Saving buildup figures...")
        for aero_component in self.aero_components:
            aero_component.save_buildup_figs()

    def load_buildup(self):
        """
        Loads the aerodynamic buildup data for all 'prime' aero components
        """
        print('Loading buildup data...')
        for aero_component in self.aero_components:
            aero_component.load_buildup()

    def generate_mesh(self):
        """
        Generate the mesh for all components and applies their local translation
        """
        for aero_component in self.aero_components:
            aero_component.generate_mesh()

    def init_actuator_dynamics(self):
        """
        TODO
        """
        aero_actuators = []
        for i in range(self.num_aero_components):
            comp_act = self.aero_components[i].actuator_model
            aero_actuators.append(comp_act)

        self.actuator_dynamics = ActuatorDynamics(aero_actuators)

    def dynamics(self, state: np.ndarray, control_deflections: np.ndarray):
        """
        Wrapper for 6-Degree of freedom dynamics in vehicle dynamics class
        """
        if self.vehicle_dynamics.allocation_matrix is None:
            raise ValueError("Vehicle dynamics is not allocated")

        cmd_deflections = self.vehicle_dynamics.allocation_matrix @ control_deflections

        true_deflections = self.actuator_dynamics.update_deflections(cmd_deflections)

        return self.vehicle_dynamics.dynamics(state, true_deflections).full().flatten()

    def get_true_deflections(self):
        """
        TODO
        """
        return self.actuator_dynamics.get_true_deflections()
