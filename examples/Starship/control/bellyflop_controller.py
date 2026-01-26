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
        # 1. Setup Variables
        state_ca = self.vehicle_dynamics.state
        m_target_ca = ca.MX.sym('m_target', 3)
        delta_ca = ca.MX.sym('delta', self.num_deflections_controllable)

        # Scaling is critical for convergence
        scale = 1e-6

        # 2. Compute Moments
        cmd_deflections = ca.DM(self.vehicle_dynamics.allocation_matrix) @ delta_ca
        _, M_symbolic = self.vehicle_dynamics.compute_aero_forces_and_moments(state_ca, cmd_deflections)

        # 3. Define Residuals (Vector of "Errors")
        # Gauss-Newton requires the objective to be in the form: sum( residuals^2 )

        # A. Moment Error (Weighted)
        # Weight Pitch (index 1) more heavily
        weights = ca.sqrt(ca.DM([1.0, 100.0, 1.0]))
        m_error = (M_symbolic - m_target_ca) * scale
        weighted_m_error = m_error * weights

        # B. Regularization (Bias to 25 deg)
        # This keeps the flaps from drifting to weird angles if they don't need to move.
        reg_weight = 1e-2
        neutral_rad = np.deg2rad(25)
        reg_error = (delta_ca - neutral_rad) * reg_weight

        # Combine into a single error vector
        all_residuals = ca.vertcat(weighted_m_error, reg_error)

        # 4. Objective: Minimize sum of squared residuals
        obj = 0.5 * ca.sumsqr(all_residuals)

        # 5. Solver Setup
        p = ca.vertcat(state_ca, m_target_ca)
        nlp = {'x': delta_ca, 'f': obj, 'p': p}

        # --- THE CRITICAL FIX ---
        opts = {
            'qpsol': 'qrqp',  # Use qrqp for the inner loop
            'qpsol_options': {
                'print_iter': False,
                'print_header': False,
                'error_on_fail': False
            },
            'print_time': False,
            'print_iteration': False,
            'max_iter': 10,

            # This flag forces J.T @ J approximation.
            # It makes the Hessian Positive Definite by definition.
            'hessian_approximation': 'gauss-newton'
        }

        # Use 'sqpmethod', NOT 'qrsqp' as the main solver
        self.solver = ca.nlpsol('S', 'sqpmethod', nlp, opts)

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
        lbx = np.full(self.num_deflections_controllable, np.deg2rad(5))
        ubx = np.full(self.num_deflections_controllable, np.deg2rad(90))

        # --- SAFETY CLAMP ---
        # Ensure the warm start is strictly within bounds before solving
        safe_x0 = np.clip(self.last_sol, lbx, ubx)

        p_val = np.concatenate((state, M_target))

        # Call solver with safe_x0
        sol = self.solver(x0=safe_x0, p=p_val, lbx=lbx, ubx=ubx)

        if self.solver.stats()['success']:
            self.last_sol = np.array(sol['x']).flatten()
        else:
            # If solver fails, hold previous valid position rather than using garbage
            # print("Solver failed, holding last position")
            pass

        return self.last_sol




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

        M_arr = np.array([0, 0, 0])
        flap_control = self.get_flap_allocation(state, M_arr)

        #flap_control = np.zeros((4, 1))
        prop_control = np.zeros((9, 1))

        return flap_control, prop_control

    def generate_wrench_set(self, state: np.ndarray) -> np.ndarray:
        angles = np.deg2rad([-20, 20])
        num_points = 100
        num_flaps = 3
        flap_angles = np.linspace(angles[0], angles[1], num_points)

        M = np.zeros((num_flaps, num_points, 3))
        for i in range(num_flaps):
            for j in range(num_points):
                control = np.zeros(4)
                control[3] = np.deg2rad(20)
                control[i] = flap_angles[j]
                deflections = self.vehicle_dynamics.allocation_matrix @ control
                F_b, M_b = self.vehicle_dynamics.compute_aero_forces_and_moments(state, deflections)
                M[i, j, :] = M_b

        return M

    def test_solve(self, state, M_target_val):
        flap_limits = np.deg2rad([5, 90])
        num_points = 100
        flap_angles = np.linspace(flap_limits[0], flap_limits[1], num_points)

        # 1. Build Interpolants
        M = self.generate_wrench_set(state)
        splines = []
        for i in range(4):
            # We simplify by creating 1D splines for each flap
            data_flat = M[i, :, :].T.flatten()
            splines.append(ca.interpolant(f'S_{i}', 'bspline', [flap_angles], data_flat))

        # 2. Define Symbolic NLP
        delta = ca.MX.sym('delta', 4)
        M_target = ca.MX.sym('M_target', 3)

        # Calculate total moment sum
        total_moment = ca.vertcat(0, 0, 0)
        for i in range(4):
            total_moment += splines[i](delta[i])

        error_cost = ca.sumsqr(total_moment - M_target)

        # Term 2: Regularization (Secondary objective)
        reg_weight = 10
        reg_cost = reg_weight * ca.sumsqr(delta - np.deg2rad([25, 25, 25, 25]))

        f = error_cost + reg_cost

        # 3. Create Solver (SQP using OSQP as the sub-problem solver)
        nlp = {'x': delta, 'f': f, 'p': M_target}
        opts = {
            'qpsol': 'qpoases',
            'hessian_approximation': 'limited-memory',  # This triggers BFGS
            'print_iteration': False,
            'print_time': False,
            'error_on_fail': False,
            'tol_pr': 1e-4,  # Relaxing primal tolerance slightly for stability
            'tol_du': 1e-4  # Relaxing dual tolerance
        }
        solver = ca.nlpsol('solver', 'sqpmethod', nlp, opts)

        # 4. Execute Solve
        # We pass the target moment as a parameter and the previous delta as a warm start
        delta_guess = np.deg2rad(np.array([25, 25, 25, 25]))
        res = solver(x0=delta_guess,
                     p=M_target_val,
                     lbx=flap_limits[0],
                     ubx=flap_limits[1])

        return np.array(res['x']).flatten()


    def draw_wrench_space(self, state: np.ndarray):
        flap_limits = np.deg2rad([5, 90])
        num_points = 100
        num_flaps = 3
        flap_angles = np.linspace(flap_limits[0], flap_limits[1], num_points)
        M = self.generate_wrench_set(state)
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
