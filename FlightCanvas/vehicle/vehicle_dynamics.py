from FlightCanvas.components.aero_component import AeroComponent

from typing import List, Union, Tuple, Optional, Dict
import numpy as np
import casadi as ca
from FlightCanvas import utils

from casadi import Function

from FlightCanvas.components.propulsion import Propulsion


class VehicleDynamics:
    """
    Vehicle dynamics class
    TODO
    """

    full_dynamics: ca.Function

    def __init__(
            self,
            mass: float,
            moi: np.ndarray,
            components: List[AeroComponent],
            control_mapping: Union[None, Dict],
            propulsion: Optional[List[Propulsion]] = None
            ):

        self.mass = mass
        self.moi = moi

        self.components = components
        self.control_mapping = control_mapping

        self.propulsion = propulsion
        self.num_propulsion = len(propulsion) if propulsion is not None else 0

        self.num_control_inputs = 0
        self.num_actuator_inputs_comp = len(components)
        if self.control_mapping is not None:
            self.num_control_inputs = len(control_mapping)
            self.allocation_matrix = self.create_allocation_matrix()

        self.create_casadi_model()

    def compute_thrust_forces_and_moments(
            self,
            true_thrust_data: Union[np.ndarray, ca.MX]
    ) -> Tuple[Union[np.ndarray, ca.MX], Union[np.ndarray, ca.MX]]:
        """
        Computes thrust forces and moments on the vehicle. This function
        is type-aware and will use either NumPy or CasADi based on the input type.
        :param true_thrust_data: This holds the current thrust and gimbal data fora all the propulsion's in the format
         defined below.
            true_thrust_data = [thrust_0, gimbal_0x, gimbal_0y, thrust_1, gimbal_1x, gimbal_1y]
        The length of the input should be 3 times the number of propulsion's
        :returns: The computed forces and moments
        """

        is_casadi = isinstance(true_thrust_data, (ca.SX, ca.MX))

        if len(true_thrust_data) % 3 != 0:
            raise ValueError("true_thrust_data must be a multiple of 3")

        if is_casadi:
            F_b = ca.MX.zeros(3, 1)
            M_b = ca.MX.zeros(3, 1)
        else:
            F_b = np.zeros(3)
            M_b = np.zeros(3)

        for i in range(len(self.propulsion)):
            propulsion = self.propulsion[i]
            thrust = true_thrust_data[3*i]
            gimbal_x = true_thrust_data[3*i + 1]
            gimbal_y = true_thrust_data[3*i + 2]
            F_b_prop, M_b_prop = propulsion.get_forces_and_moments(thrust, gimbal_x, gimbal_y)

            F_b += F_b_prop
            M_b += M_b_prop

        return F_b, M_b

    def compute_aero_forces_and_moments(
            self,
            state: Union[np.ndarray, ca.MX],
            true_deflections: Union[np.ndarray, ca.MX],
    ) -> Tuple[Union[np.ndarray, ca.MX], Union[np.ndarray, ca.MX]]:
        """
        Computes the aerodynamic forces and moments on the vehicle. This function
        is type-aware and will use either NumPy or CasADi based on the input type.
        :param state: The current state of the vehicle (position, velocity, quaternion, angular_velocity)
        :param true_deflections: The command deflection angle
        :return: The computed forces and moments
        """
        is_casadi = isinstance(state, (ca.SX, ca.MX))

        if is_casadi:
            F_b = ca.MX.zeros(3, 1)
            M_b = ca.MX.zeros(3, 1)
        else:
            F_b = np.zeros(3)
            M_b = np.zeros(3)

        # For each component, look up the forces and moments based on its local flow conditions
        for i in range(len(self.components)):
            component = self.components[i]
            true_deflection = true_deflections[i]

            F_b_comp, M_b_comp = component.get_forces_and_moments(state, true_deflection)
            F_b += F_b_comp
            M_b += M_b_comp

        return F_b, M_b

    def _calculate_rigid_body_derivatives(
            self,
            state: Union[np.ndarray, ca.MX],
            deflections_true: Union[np.ndarray, ca.MX],
            g: Union[np.ndarray, ca.MX],
            F_dist: Union[np.ndarray, ca.MX],
            M_dist: Union[np.ndarray, ca.MX]
        ) -> Tuple[Union[np.ndarray, ca.MX], Union[np.ndarray, ca.MX], Union[np.ndarray, ca.MX]]:
        """
        Calculates the rigid body derivatives given a state and deflections
        param state: The state to calculate the derivatives for
        param deflections_true: The true deflection of the flaps
        param g: The gravitational force acting on the body
        param F_dist: Force disturbances in body frame
        param M_dist: Moment disturbances in body frame
        :return: The rigid body derivatives
        """

        # Extract quaterion and angular rate from state
        quat = state[6:10]
        omega_B = state[10:13]

        # check if state is a casadi object
        is_casadi = isinstance(state, (ca.MX, ca.SX))

        # if casadi use casadi functions, else use numpy
        if is_casadi:
            inv_func = ca.inv
            cross_func = ca.cross
            array_func = ca.MX
            dir_cosine_func = utils.dir_cosine_ca
            omega_matrix_func = utils.omega_ca
        else:
            inv_func = np.linalg.inv
            cross_func = np.cross
            array_func = np.array
            dir_cosine_func = utils.dir_cosine_np
            omega_matrix_func = utils.omega

        # Calculate external forces and moments as function
        F_B_aero, M_B_aero = self.compute_aero_forces_and_moments(state, deflections_true)

        # Sum Aero and disturbances
        F_B = F_B_aero + F_dist
        M_B = M_B_aero + M_dist

        # compute direction cosine matrix
        C_B_I = dir_cosine_func(quat)   # INERTIAL frame to BODY frame.

        # Rotate Gravity into Body Frame
        g_body = C_B_I @ g

        # Body velocity
        v_body = state[3:6]

        # The Coriolis Term
        coriolis_accel = cross_func(omega_B, v_body)

        # calculate inertial acceleration
        v_dot = (F_B / self.mass) + g_body - coriolis_accel

        # get moment of inertia
        J_B = array_func(self.moi)

        # angular rate calculation
        quat_dot = 0.5 * (omega_matrix_func(omega_B) @ quat)  # + quat_dot_correction

        # angular acceleration based on conservation of momentum
        omega_dot = inv_func(J_B) @ (M_B - cross_func(omega_B, J_B @ omega_B))

        return v_dot, omega_dot, quat_dot

    def create_casadi_model(self):
        """
        TODO
        """

        # Define Symbolic State and Dynamics
        pos_I = ca.MX.sym('pos_I', 3)
        vel_I = ca.MX.sym('vel_I', 3)
        quat = ca.MX.sym('quat', 4)
        omega_B = ca.MX.sym('omega_B', 3)

        # Disturbances
        F_dist = ca.MX.sym('F_dist', 3)
        M_dist = ca.MX.sym('M_dist', 3)

        # Define Symbolic Controls
        control_deflections = ca.MX.sym('control_deflections', self.num_actuator_inputs_comp)
        g = ca.MX.sym('g', 3)

        # concat state into a single variable
        state = ca.vertcat(pos_I, vel_I, quat, omega_B)

        # calculate x_dot
        v_dot, omega_dot, quat_dot = self._calculate_rigid_body_derivatives(state, control_deflections, g, F_dist, M_dist)

        state_dot = ca.vertcat(vel_I, v_dot, quat_dot, omega_dot)

        # create casadi function of dynamics
        self.full_dynamics = Function('dynamics', [state, control_deflections, g, F_dist, M_dist], [state_dot])

    def dynamics(self, state: np.ndarray, control_inputs: np.ndarray, gravity=True):
        if self.full_dynamics is None:
            self.create_casadi_model()
        F_dist = np.array([0, 0, 0])
        M_dist = np.array([0, 0, 0])
        if gravity:
            return self.full_dynamics(state, control_inputs, np.array([0, 0, -9.81]), F_dist, M_dist)
        else:
            return self.full_dynamics(state, control_inputs, np.array([0, 0, 0]), F_dist, M_dist)

    def create_allocation_matrix(self) -> np.ndarray:
        """
        Creates the starship_control allocation matrix for the vehicle.
        """
        # Get a sorted list of high-level command names for consistent column ordering.
        command_names = self.control_mapping.keys()
        command_to_col = {name: i for i, name in enumerate(command_names)}

        comp_lookup = {comp.name: comp for comp in self.components}
        comp_lookup_by_index = {comp.name: i for i, comp in enumerate(self.components)}

        # Initialize the matrix. Rows correspond to the individual actuator inputs,
        allocation_matrix = np.zeros(
            (self.num_actuator_inputs_comp, self.num_control_inputs)
        )

        # Populate the matrix using the defined mapping.
        for command_name, component_map in self.control_mapping.items():
            # Get the column index for the current high-level command.
            col_idx = command_to_col[command_name]

            for actuator_name, gain in component_map.items():
                # Find the actuator's data (including its control_index) using its name.
                if actuator_name in comp_lookup:
                    row_idx = comp_lookup_by_index[actuator_name]
                    allocation_matrix[row_idx, col_idx] += gain
                else:
                    # Optional but recommended: A warning for names that don't match.
                    print(f"Warning: Actuator '{actuator_name}' in starship_control mapping not found in components.")

        return allocation_matrix

