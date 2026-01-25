from FlightCanvas.vehicle.vehicle_dynamics import VehicleDynamics
from FlightCanvas.control.base_controller import BaseController
from FlightCanvas.control.PID import PID
import numpy as np
import casadi as ca
from typing import Union
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
from FlightCanvas import utils


class BellyFlopController(BaseController):
    solver: ca.Function
    get_tangents: ca.Function
    dt: float

    pid_roll: PID
    pid_pitch: PID
    pid_yaw: PID

    def __init__(self, vehicle_dynamics: VehicleDynamics):
        super().__init__(vehicle_dynamics)

        self.num_deflections_controllable = self.vehicle_dynamics.allocation_matrix.shape[1]
        self.num_deflections_full = self.vehicle_dynamics.allocation_matrix.shape[0]
        self.last_sol = [np.deg2rad(25)] * self.num_deflections_controllable

    def init_solver(self):
        state_ca = self.vehicle_dynamics.state
        delta = ca.MX.sym('delta', self.num_deflections_controllable)

        # 1. Setup the Jacobian function
        cmd_deflections = ca.DM(self.vehicle_dynamics.allocation_matrix) @ delta
        _, M_symbolic = self.vehicle_dynamics.compute_aero_forces_and_moments(state_ca, cmd_deflections)

        # We output the Jacobian and the Current Moment
        self.get_tangents = ca.Function('J', [delta, state_ca], [ca.jacobian(M_symbolic, delta), M_symbolic])

        # 2. Define the QP using symbolic parameters
        d_delta = ca.MX.sym('d_delta', 4)
        H_param = ca.MX.sym('H_param', 4, 4)
        g_param = ca.MX.sym('g_param', 4)

        # Objective: 0.5 * x'Hx + g'x
        f_obj = 0.5 * ca.bilin(H_param, d_delta, d_delta) + ca.dot(g_param, d_delta)

        # Define the problem structure
        qp_struct = {
            'x': d_delta,
            'f': f_obj,
            'p': ca.vertcat(ca.reshape(H_param, -1, 1), g_param)  # Flattened H and g
        }

        opts = {'print_header': False, 'print_iter': False, 'error_on_fail': False}
        self.solver = ca.qpsol('S', 'qrqp', qp_struct, opts)
        """
        p = ca.vertcat(state_ca, M_target, last_delta_p)
        nlp = {'x': deflections_ca, 'f': obj, 'p': p}
        print_info = False
        opts = {
            'qpsol': 'qrqp',
            'print_header': False,
            'print_iteration': False,
            'max_iter': 50,  # Increase from 10 to 50
            'error_on_fail': False,  # Prevents hard crash if a frame doesn't converge
            'qpsol_options': {'print_iter': print_info, 'print_header': print_info, 'print_info': print_info,
                              'error_on_fail': False},
        }
        self.solver = ca.nlpsol('S', 'qrsqp', nlp, opts)
        """

    def init_controller(self, dt: float):
        super().init_controller(dt)
        self.init_solver()
        self.init_guidance(dt)

    def init_guidance(self, dt: float):
        Kp = 0.4
        Ki = 0
        Kd = 0.1
        tau = 1

        pid = PID(Kp, Ki, Kd, tau)
        self.pid_roll = pid.c2d(dt)
        self.pid_pitch = pid.c2d(dt)
        self.pid_yaw = pid.c2d(dt)

    def get_flap_allocation(self, state: np.ndarray, M_target: np.ndarray) -> np.ndarray:
        # 1. Get local tangents (J) and current moments (M_curr)
        quat_0 = utils.euler_to_quat((0, 0, 0))
        omega_0 = np.array([0, 0, 0])  # Initial angular velocity
        state[6:10] = quat_0
        state[10:13] = omega_0

        J_ca, M_curr_ca = self.get_tangents(self.last_sol, state)
        J = np.array(J_ca)
        M_curr = np.array(M_curr_ca).flatten()

        # --- THE FIX: Scaling/Normalization ---
        # We scale moments by 1e-6 so the solver works with small numbers
        scale = 1e-6
        M_err_scaled = (M_curr - M_target) * scale
        J_scaled = J * scale

        W_mom = 1.0  # Now that we scaled J, a weight of 1.0 is powerful
        W_center = 1e-4  # Penalty for deviating from 25 deg

        # 2. Construct H and g
        H = 2 * (W_mom * J_scaled.T @ J_scaled + W_center * np.eye(4))

        delta_25_err = self.last_sol - np.deg2rad(25)
        g = 2 * (W_mom * J_scaled.T @ M_err_scaled + W_center * delta_25_err)

        # 3. Step bounds
        lb_step = np.deg2rad(5) - self.last_sol
        ub_step = np.deg2rad(90) - self.last_sol

        # 4. Solve using the 'p' parameter
        p_values = np.concatenate([H.flatten(), g.flatten()])
        sol = self.solver(x0=np.zeros(4), p=p_values, lbx=lb_step, ubx=ub_step)

        # 5. Check for success and update
        if self.solver.stats()['success']:
            d_delta = np.array(sol['x']).flatten()
            # Apply a 'learning rate' (0.5) to prevent oscillation in highly nonlinear regions
            self.last_sol = np.clip(self.last_sol + 0.5 * d_delta, np.deg2rad(5), np.deg2rad(90))

        cmd_deflection_full = self.vehicle_dynamics.allocation_matrix @ self.last_sol
        _, M_b_true = self.vehicle_dynamics.compute_aero_forces_and_moments(state, cmd_deflection_full)
        print("M_target: ", M_target)
        print("M_b_true ", M_b_true)

        return self.last_sol

    def get_flap_allocation_OLD(self, state: np.ndarray, moments: np.ndarray) -> np.ndarray:
        lb_delta = np.array([np.deg2rad(2)] * self.num_deflections_controllable)
        ub_delta = np.array([np.deg2rad(45)] * self.num_deflections_controllable)

        p = np.concatenate((state, moments, np.array(self.last_sol).flatten()))
        sol = self.solver(x0=self.last_sol, p=p, lbx=lb_delta, ubx=ub_delta)
        if not self.solver.stats()['success']:
            # If it failed, don't use the bad data. Hold the last valid position.
            print("Warning: Solver failed. Holding last valid command.")
            return self.last_sol

        cmd_deflection = np.clip(np.array(sol['x']).flatten(), lb_delta, ub_delta)

        self.last_sol = sol['x']




        return cmd_deflection
        #prop_control = np.tile(np.array([0, 0, 0]), self.aero_vehicle.num_prop_components)
        #cmd_deflection_full = self.vehicle_dynamics.allocation_matrix @ cmd_deflection
        #x_dot_test = self.vehicle_dynamics.dynamics(state, cmd_deflection_full, prop_control)
        #_, M_b = self.vehicle_dynamics.compute_aero_forces_and_moments(state, cmd_deflection_full)

    def get_control(self, state: np.ndarray) -> Union[np.ndarray, np.ndarray]:
        quat = state[6:10]
        q0 = quat[0]
        q1 = quat[1]
        q2 = quat[2]
        q3 = quat[3]

        # MOI
        Ixx = self.vehicle_dynamics.moi[0, 0]
        Iyy = self.vehicle_dynamics.moi[1, 1]
        Izz = self.vehicle_dynamics.moi[2, 2]

        # ref
        pitch_ref = np.deg2rad(0)
        roll_ref = np.deg2rad(0)
        yaw_ref = np.deg2rad(0.1)

        quat_matrix = np.vstack((-q1, -q2, -q3, q0)).T

        # Perform vectorized conversion
        rot = R.from_quat(quat_matrix)
        euler = rot.as_euler('zyx', degrees=False).flatten()  # Returns (N, 3) matrix
        roll = euler[0]
        pitch = euler[1]
        yaw = euler[2]

        roll_err = roll_ref - roll
        pitch_err = pitch_ref - pitch
        yaw_err = yaw_ref - yaw

        roll_M_cmd = -self.pid_roll.update(roll_err) * Ixx
        pitch_M_cmd = -self.pid_pitch.update(pitch_err) * Iyy
        yaw_M_cmd = self.pid_yaw.update(yaw_err) * Izz

        M_arr = np.array([roll_M_cmd, 0, yaw_M_cmd])
        flap_control = self.get_flap_allocation(state, M_arr)

        #flap_control = np.zeros((4, 1))
        prop_control = np.zeros((9, 1))

        return flap_control, prop_control

    def draw_wrench_space(self, state: np.ndarray):
        flap_limits = np.deg2rad([0, 90])
        num_points = 100
        num_flaps = 4
        flap_angles = np.linspace(flap_limits[0], flap_limits[1], num_points)

        M = np.zeros((num_flaps, num_points, 3))
        for i in range(num_flaps):
            for j in range(num_points):
                deflections = np.zeros(5)
                deflections[i + 1] = flap_angles[j]
                deflections = deflections
                F_b, M_b = self.vehicle_dynamics.compute_aero_forces_and_moments(state, deflections)
                M[i, j, :] = M_b

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for i in range(num_flaps):
            Mx = M[i, :, 0]
            My = M[i, :, 1]
            Mz = M[i, :, 2]
            ax.plot(Mx, My, Mz)
            ax.set_xlabel('Mx (roll)')
            ax.set_ylabel('My (pitch)')
            ax.set_zlabel('Mz (yaw)')

        fig = plt.figure()
        ax = fig.add_subplot()
        flap_num = 2
        ax.plot(flap_angles, M[flap_num, :, 0])
        ax.plot(flap_angles, M[flap_num, :, 1])
        ax.plot(flap_angles, M[flap_num, :, 2])

        plt.show()
