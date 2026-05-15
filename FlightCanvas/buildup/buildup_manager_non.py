import aerosandbox as asb
import numpy as np
from matplotlib import pyplot as plt
import aerosandbox.tools.pretty_plots as p
import os
import pathlib
import pickle
from typing import Tuple
import casadi as ca
from typing import Union
from scipy.interpolate import RegularGridInterpolator


class BuildupManagerNon:
    def __init__(self, name: str,
                 vehicle_path: str,
                 aero_component: ['AeroComponent'],
                 alpha_grid_size: int = 150,
                 beta_grid_size: int = 100,
                 operating_velocity: float = 50.0
                 ):

        self.name = name
        self.vehicle_path = vehicle_path
        self.aero_component = aero_component

        self.alpha_grid_size = alpha_grid_size
        self.beta_grid_size = beta_grid_size
        self.operating_velocity = operating_velocity

        self.alpha_grid = None  # Hold grid of an angle of attack for buildup
        self.beta_grid = None  # Hold grid of sideslip angles for buildup
        self.asb_data_static = None  # To hold the aero build data
        self.aero_interpolants = None  # To hold the CasADi interpolant object
        self.aero_grid_interpolants = None

    def get_forces_and_moments(self, alpha: Union[float, ca.MX], beta: Union[float, ca.MX], speed: Union[float, ca.MX],
                               p, q, r) -> Tuple[Union[np.ndarray, ca.MX], Union[np.ndarray, ca.MX]]:
        """
        Computes forces and moments. This function is type-aware and will use
        either NumPy or CasADi based on the input type.
        """
        is_casadi = isinstance(speed, (ca.SX, ca.MX))

        if is_casadi:
            return self._get_forces_and_moments_ca(alpha, beta, speed)
        else:
            return self._get_forces_and_moments_np(alpha, beta, speed)

    def _get_forces_and_moments_np(self, alpha: float, beta: float, speed: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes aerodynamic forces and moments using 6 individual coefficient interpolations (NumPy).
        """
        if self.asb_data_static is None:
            raise RuntimeError("Aero data not available. Run compute_buildup() first.")

        # Physics Constants
        rho = 1.225
        q = 0.5 * rho * speed ** 2
        qS = q * self.aero_component.asb_airplane.s_ref
        b = self.aero_component.asb_airplane.b_ref
        c = self.aero_component.asb_airplane.c_ref

        # Helper to clean up the lookups
        query_point = [alpha, beta]

        Cl = self.aero_grid_interpolants["Cl"](query_point)[0]
        Cm = self.aero_grid_interpolants["Cm"](query_point)[0]
        Cn = self.aero_grid_interpolants["Cn"](query_point)[0]
        CA = self.aero_grid_interpolants["CA"](query_point)[0]
        CS = self.aero_grid_interpolants["CS"](query_point)[0]
        CN = self.aero_grid_interpolants["C_N"](query_point)[0]

        # Convert coefficients to physical Forces and Moments
        # F_b = [Axial, Side, Normal]
        F_b = np.array([CA, CS, CN]) * qS

        # M_b = [Roll, Pitch, Yaw]
        M_b = np.array([Cl * b, Cm * c, Cn * b]) * qS

        return F_b, M_b

    def _create_aero_interpolants(self):
        """

        Returns:

        """
        # Get grid points in radians
        alpha_lin_rad = np.deg2rad(self.alpha_grid[:, 0])
        beta_lin_rad = np.deg2rad(self.beta_grid[0, :])
        grid_axes = [alpha_lin_rad, beta_lin_rad]

        coeff_keys = ["Cl", "Cm", "Cn", "CA", "CS", "C_N"]

        sanitized_name = self.name.replace(" ", "_")
        self.aero_interpolants = {}
        self.aero_grid_interpolants = {}

        for key in coeff_keys:
            # Extract and reshape to (alpha_size, beta_size)
            data_grid = np.column_stack(self.asb_data_static[key]).reshape(self.alpha_grid.shape)

            # Transpose to align with the axis ordering (Beta, Alpha)
            data_grid_transposed = np.transpose(data_grid)
            data_flat = data_grid_transposed.ravel(order='C')

            # Create the CasADi interpolant
            self.aero_interpolants[key] = ca.interpolant(
                f'{sanitized_name}_{key}_Lookup',
                'linear',
                grid_axes,
                data_flat
            )

            self.aero_grid_interpolants[key] = RegularGridInterpolator(
                (alpha_lin_rad, beta_lin_rad),
                data_grid,
                method='linear',
            )

    def _get_coef_ca(self, alpha: ca.MX, beta: ca.MX) -> ca.MX:
        """
        TODO
        """
        if self.asb_data_static is None:
            raise RuntimeError("Aero data not available. Run compute_buildup() or load_buildup() first.")

        # If the interpolants haven't been created yet, create them now.
        if self.aero_interpolants is None:
            self._create_aero_interpolants()

        # Prepare input vector for CasADi (alpha and beta in radians)
        interp_input = ca.vertcat(alpha, beta)

        # Evaluate all 6 interpolants
        cl_m = self.aero_interpolants["Cl"](interp_input)
        cm_m = self.aero_interpolants["Cm"](interp_input)
        cn_m = self.aero_interpolants["Cn"](interp_input)
        ca_f = self.aero_interpolants["CA"](interp_input)
        cs_f = self.aero_interpolants["CS"](interp_input)
        cn_f = self.aero_interpolants["C_N"](interp_input)

        # Return as a single column vector
        return ca.vertcat(cl_m, cm_m, cn_m, ca_f, cs_f, cn_f)

    def _get_forces_and_moments_ca(self, alpha: ca.MX, beta: ca.MX, speed: ca.MX) -> tuple[ca.MX, ca.MX]:
        """
        TODO
        """
        coeffs = self._get_coef_ca(alpha, beta)

        Cl_i = coeffs[0]
        Cm_i = coeffs[1]
        Cn_i = coeffs[2]
        CA_i = coeffs[3]
        CS_i = coeffs[4]
        CN_i = coeffs[5]

        rho = 1.225
        q = 0.5 * rho * speed ** 2
        qS = q * self.aero_component.asb_airplane.s_ref
        b = self.aero_component.asb_airplane.b_ref
        c = self.aero_component.asb_airplane.c_ref

        M_b_x = Cl_i * qS * b
        M_b_y = Cm_i * qS * c
        M_b_z = Cn_i * qS * b
        F_b_x = CA_i * qS
        F_b_y = CS_i * qS
        F_b_z = CN_i * qS

        F_b = ca.vertcat(F_b_x, F_b_y, F_b_z)
        M_b = ca.vertcat(M_b_x, M_b_y, M_b_z)

        return F_b, M_b

    def compute_buildup(self):
        """
        Computes the aerodynamic buildup data for the component over a range
        of alpha and beta angles at a specified velocity
        """
        # Create a meshgrid of alpha and beta values to analyze
        self.beta_grid, self.alpha_grid = np.meshgrid(
            np.linspace(-90, 90, self.beta_grid_size),
            np.linspace(-180, 180, self.alpha_grid_size)
        )
        # Define the operating points for the analysis
        op_point = asb.OperatingPoint(
            velocity=self.operating_velocity,
            alpha=self.alpha_grid.flatten(),
            beta=self.beta_grid.flatten()
        )
        # Run the AeroBuildup analysis
        self.asb_data_static = asb.AeroBuildup(
            airplane=self.aero_component.asb_airplane,
            op_point=op_point,
        ).run()

        qS = op_point.dynamic_pressure() * self.aero_component.asb_airplane.s_ref
        self.asb_data_static["CA"] = self.asb_data_static["F_b"][0] / qS
        self.asb_data_static["CS"] = self.asb_data_static["F_b"][1] / qS
        self.asb_data_static["C_N"] = self.asb_data_static["F_b"][2] / qS

    def save_buildup(self):

        # Put variables in a dictionary for easy loading
        variables_to_save = {
            'name': self.name,
            'alpha_grid_size': self.alpha_grid_size,
            'beta_grid_size': self.beta_grid_size,
            'operating_velocity': self.operating_velocity,
            'asb_data_static': self.asb_data_static,
        }

        folder_path = os.path.join(self.vehicle_path, 'aero_data')
        path_object = pathlib.Path(folder_path)
        path_object.mkdir(parents=True, exist_ok=True)
        file_name = f"{self.name}.pkl"
        full_path = os.path.join(folder_path, file_name)

        # Open the file in binary write mode ('wb')
        with open(full_path, 'wb') as file:
            pickle.dump(variables_to_save, file)

    def load_buildup(self):
        folder_path = os.path.join(self.vehicle_path, 'aero_data')
        file_name = f"{self.name}.pkl"
        full_path = os.path.join(folder_path, file_name)

        # Check if the file exists before trying to open it
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Could not find the buildup file at: {full_path}")

        # Open the file in binary read mode ('rb') and load the data
        with open(full_path, 'rb') as file:
            loaded_data = pickle.load(file)

        # set variables form the loaded file to the current object
        self.name = loaded_data['name']
        self.alpha_grid_size = loaded_data['alpha_grid_size']
        self.beta_grid_size = loaded_data['beta_grid_size']
        self.operating_velocity = loaded_data['operating_velocity']
        self.asb_data_static = loaded_data['asb_data_static']

        self.beta_grid, self.alpha_grid = np.meshgrid(
            np.linspace(-90, 90, self.beta_grid_size),
            np.linspace(-180, 180, self.alpha_grid_size)
        )

    def save_buildup_figs(self):
        """
        Draws a contour plot of a specified aerodynamic coefficient from the
        buildup data
        """
        coeff_keys = ["Cl", "Cm", "Cn", "CA", "CS", "C_N"]

        folder_path = os.path.join(self.vehicle_path, 'buildup', self.name)
        path_object = pathlib.Path(folder_path)
        path_object.mkdir(parents=True, exist_ok=True)

        # ID is the identifier for the data to plot (e.g., "CL", "CD", "F_b")
        for key in coeff_keys:
            self.draw_buildup_figs(key, folder_path)

    def draw_buildup_figs(self, key: str, folder_path: str):
        """
        TODO
        """
        data = self.asb_data_static[key]
        title = f"`{self.name}` {key}"
        file_name = f"{self.name}_{key}.png"
        full_path = os.path.join(folder_path, file_name)

        # Create the contour plot
        plt.figure()
        p.contour(
            self.beta_grid, self.alpha_grid, data.reshape(self.alpha_grid.shape),
            colorbar_label=f"${key}$ [-]",
            linelabels_format=lambda x: f"{x:.2f}",
            linelabels_fontsize=7,
            cmap="RdBu",
            alpha=0.6
        )
        p.set_ticks(15, 5, 15, 5)
        plt.clim(*np.array([-1, 1]) * np.max(np.abs(data)))
        p.show_plot(
            title,
            r"Sideslip angle $\beta$ [deg]",
            r"Angle of Attack $\alpha$ [deg]",
            set_ticks=False,
            savefig=full_path,
            show=False
        )

        plt.close()

    def verify_np_vs_ca(self, speed: float = 50.0):
        """
        Evaluates both NumPy and CasADi functions at a test point and compares the outputs.
        Inputs for alpha and beta should be in radians to match your grid axes.
        """
        alpha_rad = np.deg2rad(120)
        beta_rad = np.deg2rad(-30)
        print(f"--- Running Verification (alpha={alpha_rad:.3f} rad, beta={beta_rad:.3f} rad, V={speed} m/s) ---")

        # Evaluate NumPy
        F_b_np, M_b_np = self._get_forces_and_moments_np(alpha_rad, beta_rad, speed)

        # We can pass CasADi DM (Data Matrix / numeric values) directly into the symbolic function
        F_b_ca_sym, M_b_ca_sym = self._get_forces_and_moments_ca(ca.DM(alpha_rad), ca.DM(beta_rad), ca.DM(speed))

        # Convert CasADi DM objects back to flat NumPy arrays for comparison
        F_b_ca = np.array(F_b_ca_sym).flatten()
        M_b_ca = np.array(M_b_ca_sym).flatten()

        # 3. Calculate Differences
        F_diff = np.abs(F_b_np - F_b_ca)
        M_diff = np.abs(M_b_np - M_b_ca)

        # 4. Print Results
        print("Forces (F_b):")
        print(f"  NumPy:  {F_b_np}")
        print(f"  CasADi: {F_b_ca}")
        print(f"  Max Diff: {np.max(F_diff):.4e}")

        print("\nMoments (M_b):")
        print(f"  NumPy:  {M_b_np}")
        print(f"  CasADi: {M_b_ca}")
        print(f"  Max Diff: {np.max(M_diff):.4e}")

        # Assertion check (using a small tolerance to account for floating point differences)
        is_match = np.allclose(F_b_np, F_b_ca, atol=1e-5) and np.allclose(M_b_np, M_b_ca, atol=1e-5)

        if is_match:
            print("\n SUCCESS: NumPy and CasADi outputs match!")
        else:
            print(
                "\n WARNING: Outputs do not match. Check interpolation types ('linear' vs 'bspline') and masking logic.")
