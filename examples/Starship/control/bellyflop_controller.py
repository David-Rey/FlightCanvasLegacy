from FlightCanvas.vehicle.aero_vehicle import AeroVehicle
import numpy as np
import casadi as ca


class BellyFlopController:
    solver: ca.Function

    def __init__(self, aero_vehicle: AeroVehicle):
        self.aero_vehicle = aero_vehicle
        self.vehicle_dynamics = self.aero_vehicle.vehicle_dynamics
        self.num_deflections_controllable = self.vehicle_dynamics.allocation_matrix.shape[1]
        self.num_deflections_full = self.vehicle_dynamics.allocation_matrix.shape[0]

    def init_solver(self):

        M_target = ca.MX.sym('M_target', 3)  # The 3x1 moment vector we want

        state_ca = self.vehicle_dynamics.state
        deflections_ca = ca.MX.sym('delta', self.num_deflections_controllable)

        cmd_deflections = ca.DM(self.vehicle_dynamics.allocation_matrix) @ deflections_ca

        _, M_b_ca = self.vehicle_dynamics.compute_aero_forces_and_moments(state_ca, cmd_deflections)

        target_flap = np.deg2rad(25)
        obj_moment = ca.norm_2((M_b_ca - M_target) / 1e6)**2
        obj_flap = ca.norm_2(target_flap - cmd_deflections)**2
        obj_reg = 1e-6 * ca.sumsqr(deflections_ca)

        W_moment = 1.0
        W_flap = 1e-4

        obj = W_moment * obj_moment + W_flap * obj_flap + obj_reg

        p = ca.vertcat(state_ca, M_target)
        nlp = {'x': deflections_ca, 'f': obj, 'p': p}
        opts = {
            'qpsol': 'qrqp',
            'print_header': False,
            'print_iteration': False,
            'max_iter': 50,  # Increase from 10 to 50
            'error_on_fail': False,  # Prevents hard crash if a frame doesn't converge
            'qpsol_options': {'print_iter': False, 'print_header': False}
        }
        self.solver = ca.nlpsol('S', 'sqpmethod', nlp, opts)

    def get_flap_allocation(self, state: np.ndarray, moments: np.ndarray) -> np.ndarray:

        lb_delta = np.array([np.deg2rad(5)] * self.num_deflections_controllable)
        ub_delta = np.array([np.deg2rad(90)] * self.num_deflections_controllable)

        p = np.concatenate((state, moments))
        sol = self.solver(x0=[np.deg2rad(25)] * self.num_deflections_controllable, p=p, lbx=lb_delta, ubx=ub_delta)

        cmd_deflection = sol['x']

        return cmd_deflection

        #prop_control = np.tile(np.array([0, 0, 0]), self.aero_vehicle.num_prop_components)
        #cmd_deflection_full = self.vehicle_dynamics.allocation_matrix @ cmd_deflection
        #x_dot_test = self.vehicle_dynamics.dynamics(state, cmd_deflection_full, prop_control)
        #_, M_b = self.vehicle_dynamics.compute_aero_forces_and_moments(state, cmd_deflection_full)



