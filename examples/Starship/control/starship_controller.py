from FlightCanvas.control.base_controller import BaseController
from FlightCanvas.vehicle.vehicle_dynamics import VehicleDynamics
from FlightCanvas.components.aero_component import AeroComponent
import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.chebyshev as cheb
from numpy.polynomial import Polynomial
import sympy as sp
from typing import Union
from FlightCanvas import utils
from FlightCanvas.control.PID import PID
from scipy.spatial.transform import Rotation as R


class StarshipController(BaseController):

    pid_roll: PID
    pid_pitch: PID
    pid_yaw: PID

    def __init__(self, vehicle_dynamics: VehicleDynamics):
        super().__init__(vehicle_dynamics)
        self.vehicle_dynamics = vehicle_dynamics

        self.num_points = 100
        self.angles = np.deg2rad([10, 90])
        self.flap_angles = np.linspace(self.angles[0], self.angles[1], self.num_points)

        self.symbolic_moment_front = {}
        self.symbolic_moment_aft = {}
        self.symbolic_moment_diff_front = {}
        self.symbolic_moment_diff_aft = {}

        #self.current_cmd_deflection = np.deg2rad([20, 20, 20, 20])
        self.current_cmd_deflection = np.array([0.36816105, 0.36816105, 0.29146177, 0.29146177])

    def init_controller(self, dt: float):
        super().init_controller(dt)

        pos_0 = np.array([0, 0, 1000])  # Initial position
        vel_0 = np.array([0, 0, -60])  # Body velocity
        quat_0 = utils.euler_to_quat((0, 0, 0))
        omega_0 = np.array([0, 0, 0])  # Initial angular velocity
        initial_state = np.concatenate((pos_0, vel_0, quat_0, omega_0))

        self.curve_fit(initial_state)
        self.init_guidance(dt)

    def perform_flap_allocation(self, state: np.ndarray, target_moment: np.ndarray, k=0.3):
        M_del_d1 = np.array([0.0, 0.0, 0.0])
        M_del_d3 = np.array([0.0, 0.0, 0.0])
        keys = ["x", "y", "z"]
        for i in range(3):
            M_del_d1[i] = self.symbolic_moment_diff_front[keys[i]](self.current_cmd_deflection[0])
            M_del_d3[i] = self.symbolic_moment_diff_aft[keys[i]](self.current_cmd_deflection[2])

        M_del_d2 = M_del_d1 * np.array([-1, 1, -1])
        M_del_d4 = M_del_d3 * np.array([-1, 1, -1])

        deflections = self.vehicle_dynamics.allocation_matrix @ self.current_cmd_deflection

        # TODO: change this to something else. I feel like using the true dynamics is cheating
        _, current_moment_true = self.vehicle_dynamics.compute_aero_forces_and_moments(state, deflections)
        current_moment = current_moment_true / self.get_dyn_pressure(state)

        #print(f"Current Moment for vehicle flaps: {current_moment}\n")

        delta_M = target_moment - current_moment

        G = np.column_stack((M_del_d1, M_del_d2, M_del_d3, M_del_d4))
        G_pinv = np.linalg.pinv(G)
        delta_u = G_pinv @ delta_M
        self.current_cmd_deflection = self.current_cmd_deflection + (delta_u * k)
        self.current_cmd_deflection = np.clip(self.current_cmd_deflection, self.angles[0], self.angles[1])

    def test_moment(self, state: np.ndarray):
        fuse_comp = self.vehicle_dynamics.aero_components[0]
        _, moment_offset = self.nondim_force_and_moment(state, 0, fuse_comp)

        target_moment = np.array([0.0, 0.0, 0.0])
        print(f"Target Moment for 4 flaps: {target_moment}\n")

        iters = 12
        k = 0.3
        for i in range(iters):
            print(f"Iteration {i}")
            print(f"Flap position: {self.current_cmd_deflection}")

            M_del_d1 = np.array([0.0, 0.0, 0.0])
            M_del_d3 = np.array([0.0, 0.0, 0.0])
            keys = ["x", "y", "z"]
            for i in range(3):
                M_del_d1[i] = self.symbolic_moment_diff_front[keys[i]](self.current_cmd_deflection[0])
                M_del_d3[i] = self.symbolic_moment_diff_aft[keys[i]](self.current_cmd_deflection[2])

            M_del_d2 = M_del_d1 * np.array([-1, 1, -1])
            M_del_d4 = M_del_d3 * np.array([-1, 1, -1])

            deflections = self.vehicle_dynamics.allocation_matrix @ self.current_cmd_deflection
            _, current_moment_true = self.vehicle_dynamics.compute_aero_forces_and_moments(state, deflections)
            current_moment = current_moment_true / self.get_dyn_pressure(state)

            print(f"Current Moment for vehicle flaps: {current_moment}\n")

            delta_M = target_moment - current_moment

            G = np.column_stack((M_del_d1, M_del_d2, M_del_d3, M_del_d4))
            G_pinv = np.linalg.pinv(G)
            delta_u = G_pinv @ delta_M
            self.current_cmd_deflection = self.current_cmd_deflection + (delta_u * k)

    def curve_fit(self, state: np.ndarray):

        comp_front = self.vehicle_dynamics.aero_components[1]
        comp_aft = self.vehicle_dynamics.aero_components[3]
        symbolic_moment_front_sp, symbolic_moment_diff_front_sp, M_1 = self.get_symbolic_fit(state, comp_front)
        symbolic_moment_aft_sp, symbolic_moment_diff_aft_sp, M_3 = self.get_symbolic_fit(state, comp_aft)

        delta = sp.Symbol('delta')
        for key in ["x", "y", "z"]:
            self.symbolic_moment_front[key] = sp.lambdify(delta, symbolic_moment_front_sp[key], 'numpy')
            self.symbolic_moment_aft[key] = sp.lambdify(delta, symbolic_moment_aft_sp[key], 'numpy')
            self.symbolic_moment_diff_front[key] = sp.lambdify(delta, symbolic_moment_diff_front_sp[key], 'numpy')
            self.symbolic_moment_diff_aft[key] = sp.lambdify(delta, symbolic_moment_diff_aft_sp[key], 'numpy')

        '''
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 4), sharex=True)
        ax1.plot(self.flap_angles, M_1[:, 0], color='red')
        ax1.plot(self.flap_angles, M_1[:, 1], color='green')
        ax1.plot(self.flap_angles, M_1[:, 2], color='blue')
        ax1.set_ylabel('Moment (N-m)')
        ax1.set_xlabel('Angle (rad)')
        ax1.grid(True)

        ax2.plot(self.flap_angles, M_3[:, 0], color='red')
        ax2.plot(self.flap_angles, M_3[:, 1], color='green')
        ax2.plot(self.flap_angles, M_3[:, 2], color='blue')
        ax2.set_ylabel('Moment (N-m)')
        ax2.set_xlabel('Angle (rad)')
        ax2.grid(True)
        '''

    def get_symbolic_fit(self, state: np.ndarray, comp: AeroComponent) -> tuple[dict, dict, np.ndarray]:

        M_1 = np.zeros((self.num_points, 3))
        for i in range(self.num_points):
            _, M_1[i, :] = self.nondim_force_and_moment(state, self.flap_angles[i], comp)

        angle_min = self.flap_angles.min()
        angle_max = self.flap_angles.max()
        angles_norm = (2.0 * self.flap_angles - (angle_max + angle_min)) / (angle_max - angle_min)
        degree = 4
        delta = sp.Symbol('delta')

        symbolic_moment = {}
        symbolic_moment_diff = {}
        keys = ["x", "y", "z"]

        for i in range(3):
            c_coeffs = cheb.chebfit(angles_norm, M_1[:, i], degree)
            cheb_series = cheb.Chebyshev(c_coeffs, domain=[angle_min, angle_max])
            poly_series = cheb_series.convert(kind=Polynomial)
            std_coeffs = poly_series.coef

            # Construct the symbolic polynomial
            express = sum(c * delta ** i for i, c in enumerate(std_coeffs))
            symbolic_moment[keys[i]] = express
            symbolic_moment_diff[keys[i]] = sp.diff(express, delta)

        return symbolic_moment, symbolic_moment_diff, M_1

    def draw_wrench(self, state: np.ndarray):
        num_points = 100
        angles = np.deg2rad([5, 90])
        flap_angles = np.linspace(angles[0], angles[1], num_points)
        M_1 = np.zeros((num_points, 3))
        comp = self.vehicle_dynamics.aero_components[1]

        for i in range(num_points):
            _, M_1[i, :] = self.nondim_force_and_moment(state, flap_angles[i], comp)

        fig = plt.figure()
        ax = fig.add_subplot()
        plt.plot(flap_angles, M_1[:, 0])
        plt.plot(flap_angles, M_1[:, 1])
        plt.plot(flap_angles, M_1[:, 2])
        ax.set_xlabel('angle')
        ax.set_ylabel('moment')
        plt.show()

    def get_dyn_pressure(self, state: np.ndarray) -> float:
        rho = 1.225
        speed = np.linalg.norm(state[3:6])
        q = 0.5 * rho * speed ** 2
        return q

    def nondim_force_and_moment(self, state: np.ndarray, flap_angle: float, comp: AeroComponent):
        q = self.get_dyn_pressure(state)
        F_b, M_b = comp.get_forces_and_moments(state, flap_angle)
        return F_b / q, M_b / q

    def draw_3d_wrench_space(self, state: np.ndarray):
        default_control = np.deg2rad(np.array([0, 0, 0, 0]))
        q = self.get_dyn_pressure(state)

        angles = np.deg2rad([5, 90])
        num_points = 100
        num_flaps = 4
        flap_angles = np.linspace(angles[0], angles[1], num_points)

        M = np.zeros((num_flaps, num_points, 3))
        for i in range(num_flaps):
            for j in range(num_points):
                control = np.zeros(4)

                control += default_control
                control[i] = flap_angles[j]
                deflections = self.vehicle_dynamics.allocation_matrix @ control
                F_b, M_b = self.vehicle_dynamics.compute_aero_forces_and_moments(state, deflections)
                M[i, j, :] = M_b / q

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
        ax.scatter(0, 0, 0, color='red', s=20, marker='o')
        plt.show()

    def init_guidance(self, dt: float):
        Kp = 0.4
        Ki = 0
        Kd = 0.1
        tau = 1

        pid = PID(Kp, Ki, Kd, tau)
        self.pid_roll = pid.c2d(dt)
        self.pid_pitch = pid.c2d(dt)
        self.pid_yaw = pid.c2d(dt)

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
        yaw_ref = np.deg2rad(0)

        quat_matrix = np.vstack((-q1, -q2, -q3, q0)).T

        # Perform vectorized conversion
        rot = R.from_quat(quat_matrix)
        euler = rot.as_euler('zyx', degrees=False).flatten()  # Returns (N, 3) matrix
        roll = euler[2]
        pitch = euler[1]
        yaw = euler[0]

        roll_err = roll_ref - roll
        pitch_err = pitch_ref - pitch
        yaw_err = yaw_ref - yaw

        roll_M_cmd = -self.pid_roll.update(roll_err) * Ixx
        pitch_M_cmd = -self.pid_pitch.update(pitch_err) * Iyy
        yaw_M_cmd = -self.pid_yaw.update(yaw_err) * Izz

        target_moment_true = np.array([roll_M_cmd, pitch_M_cmd, yaw_M_cmd])
        q = self.get_dyn_pressure(state)

        moment_max = np.array([100, 300, 600])
        moment_min = -moment_max
        target_moment = np.clip(target_moment_true / q, moment_min, moment_max)


        prop_control = np.zeros((9, 1))
        self.perform_flap_allocation(state, target_moment)
        return self.current_cmd_deflection, prop_control




