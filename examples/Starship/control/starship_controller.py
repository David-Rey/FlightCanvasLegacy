from FlightCanvas.vehicle.vehicle_dynamics import VehicleDynamics
import numpy as np
import matplotlib.pyplot as plt


class StarshipController:
    def __init__(self, vehicle_dynamics: VehicleDynamics):
        self.vehicle_dynamics = vehicle_dynamics


    def draw_wrench_space(self, state: np.ndarray):
        default_control = np.deg2rad(np.array([0, 0, 0, 0]))
        #true_def = self.vehicle_dynamics.allocation_matrix @ flap_def

        angles = np.deg2rad([5, 55])
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
                #F_b, M_b = self.vehicle_dynamics.compute_aero_forces_and_moments(state, deflections)
                F_b, M_b = self.vehicle_dynamics.aero_components[i+1].get_forces_and_moments(state, flap_angles[j])
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
        ax.scatter(0, 0, 0, color='red', s=20, marker='o')
        plt.show()

    def get_control(self, state: np.ndarray) -> np.ndarray:
        pass
