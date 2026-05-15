import aerosandbox as asb
import aerosandbox.numpy as np
from scipy.interpolate import splprep, splev

from FlightCanvas.vehicle.aero_vehicle import AeroVehicle
from FlightCanvas.components.aero_fuselage import AeroFuselage
from FlightCanvas.components.propulsion import Propulsion
from FlightCanvas.components.aero_wing import create_planar_wing_pair, AeroWing
from FlightCanvas.Flight.flight import Flight
from FlightCanvas.analysis.log import Log
from FlightCanvas.analysis.anlaysis import Analysis
from FlightCanvas.analysis.vehicle_visualizer import VehicleVisualizer
from examples.Starship.control.starship_controller import StarshipController
from FlightCanvas.control.siso_system import SISOSystem
from FlightCanvas import utils
from scipy.ndimage import gaussian_filter
from typing import Dict, List


class Starship:
    """
    A class that defines and simulates a Starship-like aero vehicle.

    This class encapsulates the geometry, aerodynamics, mass properties,
    and starship_control systems for the vehicle, and provides methods to run
    simulations and visualizations.
    """

    def __init__(self, cg_x=25, height=50.0, diameter=9.0):
        """
        Initializes and builds the Starship vehicle model
        :param cg_x: The initial cg in x direction in m
        :param height: The height in m
        :param diameter: The diameter in m
        """
        # Store geometric parameters
        self.cg_x = cg_x
        self.height = height
        self.diameter = diameter

        # Rocket engine
        self.max_thrust = 1e6
        self.min_thrust = 0.4 * self.max_thrust
        self.gimbal_limits = 14
        self.engine_size = 1.2

        # Create all geometric components
        body = self._create_body()
        front_flaps = self._create_front_flaps()
        back_flaps = self._create_back_flaps()
        prop = self._create_rocket_engine()

        all_aero_components = [body, *front_flaps, *back_flaps]

        # Define the starship_control mapping
        control_mapping = self._get_control_mapping()

        # Assemble the AeroVehicle
        self.vehicle = AeroVehicle(
            name="Starship",
            xyz_ref=[0, 0, 0],
            aero_components=all_aero_components,
            prop_components=prop
        )

        # Set mass and inertia properties
        self._set_mass_properties()

        # Load pre-computed aerodynamic data
        print("Loading aerodynamic buildup data...")
        try:
            self.vehicle.compute_buildup()
        except:
            print("Build up data not found. Creating new aerodynamic buildup data...")
            self.vehicle.compute_buildup()
            self.vehicle.save_buildup()

        self.update_moment()

        # Init vehicle dynamics
        self.vehicle.init_vehicle_dynamics(control_mapping)

    def save_buildup(self):
        """
        Saves the aerodynamic buildup data
        """
        self.vehicle.save_buildup()

    def save_buildup_figs(self):
        """
        Saves the aerodynamic buildup figures
        """
        self.vehicle.save_buildup_fig()

    def _set_mass_properties(self):
        """
        Sets the mass and moment of inertia for the vehicle
        """
        self.vehicle.set_mass(135000)
        # MOI for a cylinder
        radius = self.diameter / 2
        I_s = (1 / 2) * radius ** 2  # Inertia about the spin axis (x)
        I_a = ((1 / 4) * radius ** 2) + ((1 / 12) * self.height ** 2)  # Inertia about transverse axes (y, z)
        self.vehicle.set_moi_diag([I_s, I_a, I_a])

    def update_moment(self):
        """
        Updates the body moments to get the cg to be reasonable
        """
        CA_temp = self.vehicle.components[0].buildup_manager.asb_data_static["Cm"]
        self.vehicle.components[0].buildup_manager.asb_data_static["Cm"] = CA_temp * 0.4

        CA_temp = self.vehicle.components[0].buildup_manager.asb_data_static["Cn"]
        self.vehicle.components[0].buildup_manager.asb_data_static["Cn"] = CA_temp * 0

        for i in [1, 3]:
            for key in ["CA", "Cl", "Cn", "CS"]:
                temp = self.vehicle.components[i].buildup_manager.asb_data_static[key]
                self.vehicle.components[i].buildup_manager.asb_data_static[key] = temp * 0

            # add smoothing for front and aft flap
            for key in ["C_N", "Cm"]:
                temp_Cm = self.vehicle.components[i].buildup_manager.asb_data_static[key]
                temp_Cm_2d = np.reshape(temp_Cm, self.vehicle.components[i].buildup_manager.alpha_grid.shape)
                smoothed_Cm_2d = gaussian_filter(temp_Cm_2d, sigma=4.0)
                smoothed_Cm_1d = smoothed_Cm_2d.flatten()
                self.vehicle.components[i].buildup_manager.asb_data_static[key] = smoothed_Cm_1d

    def _create_body(self) -> AeroFuselage:
        """
        Creates the body of starship
        :return: AeroFuselage object of starship body
        """
        n_points = 100
        nosecone_coords = self._get_nosecone_cords(self.diameter, n_points=n_points)
        end_cord = np.array([[self.height, nosecone_coords[-1, 1]], [self.height, 0]])
        nosecone_coords = np.vstack((nosecone_coords, end_cord))

        fuselage_xsecs = [
            asb.FuselageXSec(
                xyz_c=[x - self.cg_x, 0, 0],
                radius=z,
            ) for x, z in nosecone_coords
        ]

        return AeroFuselage(
            name="Fuselage",
            xsecs=fuselage_xsecs,
        )

    def _create_front_flaps(self) -> List[AeroWing]:
        """
        Creates a pair of front flap components
        """
        flap_airfoil = asb.Airfoil(coordinates=self._flat_plate_airfoil(thickness=0.02))
        front_flap_xsecs = [
            asb.WingXSec(xyz_le=[0, 0, 0], chord=8, airfoil=flap_airfoil),
            asb.WingXSec(xyz_le=[6, 4.8, 0], chord=2.5, airfoil=flap_airfoil)
        ]
        return create_planar_wing_pair(
            name="Front Flap",
            xsecs=front_flap_xsecs,
            translation=[5 - self.cg_x, 2.9, 0],
            ref_direction=[1, 0.18, 0],
            control_pivot=[1, 0.18, 0],
            actuator_model=SISOSystem([1], [0.1, 1])
        )

    def _create_back_flaps(self) -> List[AeroWing]:
        """
        Creates the pair of aft flap components
        """
        flap_airfoil = asb.Airfoil(coordinates=self._flat_plate_airfoil(thickness=0.02))
        back_flap_xsecs = [
            asb.WingXSec(xyz_le=[0, 0, 0], chord=15, airfoil=flap_airfoil),
            asb.WingXSec(xyz_le=[8, 5.8, 0], chord=6, airfoil=flap_airfoil)
        ]
        return create_planar_wing_pair(
            name="Aft Flap",
            xsecs=back_flap_xsecs,
            translation=[35 - self.cg_x, 4.5, 0],
            ref_direction=[1, 0, 0],
            control_pivot=[1, 0, 0],
            actuator_model=SISOSystem([1], [0.1, 1])
        )

    def _create_rocket_engine(self) -> List[Propulsion]:
        """
        Create rocket engine components
        """
        num_engines = 3
        spacing = 1
        angles = np.linspace(0, 2 * np.pi, num_engines + 1)[:-1]
        props = []
        for angle in angles:
            y_ref = spacing * np.sin(angle)
            z_ref = spacing * np.cos(angle)
            prop = Propulsion(
                name="Rocket Engine",
                thrust_direction=[1, 0, 0],
                thrust_bounds=[self.min_thrust, self.max_thrust],
                gimbal_bounds=[-self.gimbal_limits, self.gimbal_limits],
                size=self.engine_size,
                xyz_ref=[50 - self.cg_x, y_ref, z_ref]
            )
            props.append(prop)
        return props

    @staticmethod
    def _get_control_mapping() -> Dict[str, Dict[str, float]]:
        """
        Define how abstract starship_control commands map to individual flap deflections
        :return: Control mapping from abstract commands to flap deflections
        """
        return {
            "fl": {
                "Front Flap": 1.0,
            },
            "fr": {
                "Front Flap Star": 1.0,
            },
            "al": {
                "Aft Flap": 1.0,
            },
            "af": {
                "Aft Flap Star": 1.0
            }
        }

    # Helper methods for geometry creation
    @staticmethod
    def _smooth_path(points: np.ndarray, smoothing_factor: float = 0.0, n_points: int = 500) -> np.ndarray:
        """
        Create smooth spline interpolation through given points
        """
        tck, u = splprep(points.T, s=smoothing_factor)
        u_fine = np.linspace(0, 1, n_points)
        return np.array(splev(u_fine, tck)).T

    def _get_nosecone_cords(self, diameter, smoothed=True, n_points=500) -> np.ndarray:
        """
        Generate nosecone profile coordinates based on Starship-like proportions
        :param diameter: The diameter of the starship
        :param smoothed: If true, smooth profile coordinates
        :param n_points: Number of points in final profile
        :return: The nosecone profile coordinates
        """
        points = np.array([
            [0.010000, 0.000000], [0.057585, 0.238814], [0.286398, 0.495763],
            [2.231314, 1.601695], [3.222839, 2.097458], [6.502500, 3.394068],
            [10.697415, 4.309322], [14.320297, 4.500000]
        ])
        scaled_points = np.copy(points)
        scaled_points[:, 1] *= diameter / 4.5 / 2
        return self._smooth_path(scaled_points, smoothing_factor=0.002,
                                 n_points=n_points) if smoothed else scaled_points

    @staticmethod
    def _flat_plate_airfoil(thickness=0.01, n_points=100) -> np.ndarray:
        """
        Generate flat plate airfoil coordinates for starship_control surfaces
        """
        x = np.linspace(1, 0, n_points)
        y_upper = thickness / 2 * np.ones_like(x)
        y_lower = -thickness / 2 * np.ones_like(x)
        x_coords = np.concatenate([x, x[::-1]])
        y_coords = np.concatenate([y_upper, y_lower[::-1]])
        return np.vstack([x_coords, y_coords]).T

    def run_sim(self):
        """
        Runs static simulation
        """
        state_names = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'q0', 'q1', 'q2', 'q3', 'wx', 'wy', 'wz']
        control_names = ['pitch', 'yaw', 'roll', 'drag']
        deflection_names = ['b', 'f_left', 'f_right', 'b_left', 'b_right']
        num_engines = 3
        maxSteps = 2000
        log = Log(state_names, control_names, deflection_names, num_engines, maxSteps)

        dt = 0.01
        tf = 10

        #z_guess = np.array([-60, 0, np.deg2rad(12)])
        #trim.get_trimpoint(z_guess)
        #trim.init_LQR()
        #trim.yaw_control_analysis()

        #trim.plot_pz_with_feedback()

        #initial_state = trim.x_star

        #inital_state = trim.u_star
        #starship_control = StarshipController(sys)
        #starship_control.compute_lqr()
        #inital_state, inital_control, _ = trim.get_trimpoint()
        #trim.get_LQR_control()

        #initial_state[2] = 800
        #initial_state[11] = 0.001
        pos_0 = np.array([0, 0, 1000])  # Initial position
        vel_0 = np.array([0, 0, -60])  # Body velocity
        quat_0 = utils.euler_to_quat((0, 0, 0))
        omega_0 = np.array([0, 0, 0])  # Initial angular velocity
        initial_state = np.concatenate((pos_0, vel_0, quat_0, omega_0))

        controller = StarshipController(self.vehicle.vehicle_dynamics)
        #controller.draw_wrench_space(initial_state)
        controller.curve_fit(initial_state)
        controller.test_moment(initial_state)
        controller.draw_3d_wrench_space(initial_state)
        #controller.init_controller(dt)
        #controller.get_control(initial_state)
        #controller.debug_control_authority(initial_state, np.array([0, 0, 0]))


        #flight = Flight(self.vehicle, controller, tf, dt=dt)

        #M_b = np.array([0, 3000, 0])
        #con.get_flap_allocation(initial_state, M_b)

        #trim.draw_wrench_space(initial_state)

        #flight.run_sim(initial_state, log)

        #analysis = Analysis(log)
        #analysis.generate_control_plot()
        #analysis.generate_velocity_plot(include_vz=False)
        #analysis.generate_position_plot()
        #analysis.generate_euler_angle_plot()
        #analysis.generate_angular_velocity_plot()
        #analysis.generate_quat_norm_plot()
        #analysis.generate_angle_of_attack_plot()
        #analysis.generate_true_deflections_plot()

        #vv = VehicleVisualizer(self.vehicle)
        #vv.init_aero_actors(opacity=1)
        #vv.init_prop_actors(opacity=1, color='grey')
        #vv.init_debug(size=4)
        #vv.show()
        #vv.add_grid()
        #self.vehicle.prop_components[0].update_dynamic_transform(initial_state)

        #vv.update_actors(initial_state, np.array([0, 0, 0, 0, 0]), np.array([1, 0, 0]))
        #vv.generate_square_traj()
        #vv.show()
        #vv.animate(log, cam_distance=80, zoom=1.5)

if __name__ == '__main__':
    # Create an instance of the entire Starship model
    starship = Starship()
    #starship.save_buildup_figs()

    starship.run_sim()

    #vv = VehicleVisualizer(starship.vehicle)
    #vv.init_aero_actors(opacity=.5)
    #vv.init_prop_actors(opacity=1, color='grey')
    #vv.init_debug(size=4)
    #vv.show()

