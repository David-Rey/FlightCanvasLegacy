from FlightCanvas.vehicle.vehicle_dynamics import VehicleDynamics
import numpy as np
import casadi as ca


class BellyFlopController:
    def __init__(self, vehicle_dynamics: VehicleDynamics):
        self.vehicle_dynamics = vehicle_dynamics

        M_target = ca.MX.sym('M_target', 3)  # The 3x1 moment vector we want

        state_ca = self.vehicle_dynamics.state
        deflections_ca = self.vehicle_dynamics.aero_control_deflections

        cmd_deflections = ca.DM(self.vehicle_dynamics.allocation_matrix.T) @ deflections_ca

        _, M_b_ca = self.vehicle_dynamics.compute_aero_forces_and_moments(state_ca, deflections_ca)

        obj = ca.norm_2(M_b_ca - M_target)**2

        nlp = {'x': deflections_ca, 'f': obj, 'p': state_ca}
        opts = {
            'qpsol': 'qrqp',  # A fast, built-in dense QP solver
            'print_header': False,
            'print_iteration': False,
            'max_iter': 10  # You can limit iterations for real-time performance
        }
        self.solver = ca.nlpsol('S', 'sqpmethod', nlp, opts)
        self.num_deflections = deflections_ca.shape[0]

    def get_flap_allocation(self, state: np.ndarray, moments: np.ndarray) -> np.ndarray:
        lb_delta = [-0.349] * n_flaps
        ub_delta = [0.349] * n_flaps
        sol = self.solver(x0=[0] * n_flaps, p=p_val, lbx=lb_delta, ubx=ub_delta)

        self.vehicle_dynamics.dynamics(state, moments)
