import numpy as np
import pyvista as pv
import casadi as ca

from typing import Union, List, Tuple

from FlightCanvas.components.component import Component
from FlightCanvas import utils


class Propulsion(Component):
    def __init__(
            self,
            name: str,
            thrust_direction: Union[np.ndarray, List[float]],
            thrust_bounds: Union[np.ndarray, List[float]],
            gimbal_bounds: Union[np.ndarray, List[float]],
            size=1.0,
            x_direction=np.array([0., 1., 0.]),
            xyz_ref=np.array([0., 0., 0.])
    ):

        super().__init__(name, xyz_ref)
        self.thrust_direction = np.array(thrust_direction)
        self.thrust_bounds = np.array(thrust_bounds)
        self.gimbal_bounds = np.array(gimbal_bounds)
        self.size = size
        self.x_direction = np.array(x_direction)

        self.plume_length = 25.0
        self.n_cones = 0
        self.n_spheres = 0
        self.n_cylinders = 10

        self.nozzle_actor = None
        self.nozzle_exit_actor = None
        self.plume_meshes = None

    def translate(self, xyz: Union[np.ndarray, List[float]]) -> "Propulsion":
        """
        Sets the component's reference position relative to the vehicle's origin
        :param xyz: The new reference position [x, y, z] in the vehicle's body frame
        :return: The instance of the component (`self`)
        """
        self.xyz_ref = np.array(xyz)
        return self

    def init_actor(self, pl: pv.Plotter, **kwargs):
        """
        Adds the component's mesh to a PyVista plotter to create a renderable actor
        :param pl: The `pyvista.Plotter` instance to add the mesh to
        :param kwargs: Additional keyword arguments to pass to `pl.add_mesh()`
        """
        nozzle_exit = pv.Circle(radius=self.size / 2, resolution=36)
        self.nozzle_exit_actor = pl.add_mesh(nozzle_exit, color='black', line_width=3, style='wireframe')

        # Add rocket nozzle cylinder
        nozzle_cylinder = pv.Cylinder(
            center=(0, 0, 0),
            direction=(0, 0, 1),
            radius=self.size / 2,
            height=self.size / 2,
            resolution=32
        )
        self.nozzle_actor = pl.add_mesh(nozzle_cylinder, **kwargs)
        transform = self.get_transform()

        self.nozzle_actor.user_matrix = transform
        self.nozzle_exit_actor.user_matrix = transform

        #reflection_x = np.eye(4)
        #reflection_x[0, 0] = -1

        self.plume_meshes = self.create_plume_geometry()
        for mesh_data in self.plume_meshes:
            mesh_data['actor'] = pl.add_mesh(
                mesh_data['mesh'],
                color=mesh_data['color'],
                opacity=mesh_data['opacity'],
                smooth_shading=True
            )
            mesh_data['actor'].user_matrix = transform #@ reflection_x

        #gimbal_x = 0
        #gimbal_y = 0
        #self.update_plume_transform(np.array([1, gimbal_x, gimbal_y]))
        #self.get_forces_and_moments(np.array([1, gimbal_x, gimbal_y]))

    def init_debug(self, pl: pv.Plotter, com: np.ndarray, size: float = 1.0, label=True):
        default_sphere_radius = 0.02
        sphere_radius = default_sphere_radius * 2 ** (size - 1)

        # Draw a sphere at the component's reference point
        sphere = pv.Sphere(radius=sphere_radius, center=self.xyz_ref)
        pl.add_mesh(sphere, color='grey', show_edges=False)

        # Draw an arrow from the component's ref point to the vehicle's CoM
        utils.plot_line_from_points(pl, self.xyz_ref, com, color='grey')

        axis_length = self.size
        transform = self.get_transform()

        # Origin point in local frame
        origin = np.array([0, 0, 0, 1])
        origin_world = transform @ origin
        origin_world = origin_world[:3]

        # Axis endpoints in local frame
        x_end = np.array([axis_length, 0, 0, 1])
        y_end = np.array([0, axis_length, 0, 1])
        z_end = np.array([0, 0, axis_length, 1])

        # Transform to world frame
        x_end_world = (transform @ x_end)[:3]
        y_end_world = (transform @ y_end)[:3]
        z_end_world = (transform @ z_end)[:3]

        # X axis
        x_arrow = pv.Arrow(start=origin_world, direction=x_end_world - origin_world,
                           scale=np.linalg.norm(x_end_world - origin_world))
        pl.add_mesh(x_arrow, color='red', opacity=0.8)

        # Y axis
        y_arrow = pv.Arrow(start=origin_world, direction=y_end_world - origin_world,
                           scale=np.linalg.norm(y_end_world - origin_world))
        pl.add_mesh(y_arrow, color='green', opacity=0.8)

        # Z axis
        z_arrow = pv.Arrow(start=origin_world, direction=z_end_world - origin_world,
                           scale=np.linalg.norm(z_end_world - origin_world))
        pl.add_mesh(z_arrow, color='blue', opacity=0.8)

    def update_actor(self, state: np.ndarray, thrust_state: np.ndarray):
        self.static_transform_matrix = self.get_transform()
        self.update_dynamic_transform(state)

        self.nozzle_actor.user_matrix = self.dynamic_transform_matrix
        self.nozzle_exit_actor.user_matrix = self.dynamic_transform_matrix

        reflection_z = np.eye(4)
        reflection_z[2, 2] = -1

        for mesh_data in self.plume_meshes:
            mesh_data['actor'].user_matrix = self.dynamic_transform_matrix @ reflection_z

        self.update_plume_transform(thrust_state)

    def get_transform(self, **kwargs) -> np.ndarray:
        """
        Calculates the 4x4 homogeneous transformation matrix for the component
        :return: The 4x4 transformation matrix
        """
        transform_from_ref = utils.translation_matrix(self.xyz_ref)
        rotation_transformation_matrix = np.eye(4)

        u = self.thrust_direction / np.linalg.norm(self.thrust_direction)

        w_raw = np.cross(u, self.x_direction)
        w = w_raw / np.linalg.norm(w_raw)
        v = np.cross(w, u)
        R = np.column_stack((v, w, u))
        rotation_transformation_matrix[:3, :3] = R

        final_transform = (
                transform_from_ref @
                rotation_transformation_matrix
        )

        return final_transform

    def update_transform(self, **kwargs):
        """
        Computes and updates the component's stored transformation matrix
        :param kwargs: Additional keyword arguments to pass to get_transform
        """
        self.static_transform_matrix = self.get_transform(**kwargs)

    def get_forces_and_moments(
            self,
            thrust_state: Union[np.ndarray, ca.MX],
    ) -> Tuple[Union[np.ndarray, ca.MX], Union[np.ndarray, ca.MX]]:
        """
        TODO
        """
        is_casadi = isinstance(thrust_state, (ca.SX, ca.MX))

        # Select the correct library functions and types based on the input
        if is_casadi:
            lib = ca
            to_type = ca.MX
        else:
            lib = np
            to_type = lambda x: x

        # Get transform matrix
        T = to_type(self.get_transform())

        # Get rotation matrix from body frame to component frame
        R = T[:3, :3]

        # Compute distance from component to vehicle center of mass in the body frame
        lever_arm = to_type(self.parent.xyz_ref - self.xyz_ref)

        # Get thrust and gimbal angles
        thrust = thrust_state[0] * self.thrust_bounds[1]

        # Gets rotation matrix from component frame to gimbal frame
        R_g = self.get_gimbal_transform(thrust_state)[:3, :3]

        # Get thrust in gimbal frame
        if is_casadi:
            F_g = ca.vertcat(0, 0, thrust)
        else:
            F_g = np.array([0, 0, thrust])

        # Convert thrust from gimbal frame to component frame
        F_c = R_g @ F_g

        # Convert thrust from component frame to body frame
        F_b = R @ F_c

        # Compute thrust moment
        M_b = lib.cross(lever_arm, F_b)

        return F_b, M_b

    def create_plume_geometry(self):
        """
        Create all plume meshes at full throttle
        """
        colors_hot = np.array([
            [1.0, 1.0, 0.6],  # Light yellow
            [1.0, 0.9, 0.3],  # Yellow
            [1.0, 0.7, 0.2],  # Orange-yellow
            [1.0, 0.5, 0.1],  # Orange
            [1.0, 0.3, 0.0],  # Red-orange
            [0.8, 0.2, 0.0],  # Dark red
        ])

        meshes = []

        # Create cones
        for i in range(self.n_cones):
            t = i / self.n_cones
            z_pos = t * self.plume_length
            base_radius = self.size / 2
            expansion = 0.4 * t
            radius = base_radius + expansion
            cone_height = 10 / self.n_cones * 1.2

            cone = pv.Cone(
                center=(0, 0, z_pos - cone_height / 2),
                direction=(0, 0, -1),
                height=cone_height,
                radius=radius,
                resolution=32
            )

            color_idx = min(int(t * (len(colors_hot) - 1)), len(colors_hot) - 1)
            color_t = (t * (len(colors_hot) - 1)) - color_idx

            if color_idx < len(colors_hot) - 1:
                color = colors_hot[color_idx] * (1 - color_t) + colors_hot[color_idx + 1] * color_t
            else:
                color = colors_hot[color_idx]

            opacity = 0.4 * (1 - t * 0.5)

            meshes.append({
                'mesh': cone,
                'original_mesh': cone.copy(),  # Store original
                'type': 'cone',
                'color': color,
                'opacity': opacity,
                't': t,
                'base_z': z_pos,
                'base_radius': radius,
                'height': cone_height
            })

        # Create spheres
        for i in range(self.n_spheres):
            t = i / self.n_spheres
            z_pos = t * self.plume_length
            base_radius = self.size / 2
            expansion = 0.4 * t
            sphere_radius = (base_radius + expansion) * 0.8

            sphere = pv.Sphere(radius=sphere_radius, center=(0, 0, z_pos),
                               phi_resolution=20, theta_resolution=20)

            color_idx = min(int(t * (len(colors_hot) - 1)), len(colors_hot) - 1)
            color_t = (t * (len(colors_hot) - 1)) - color_idx

            if color_idx < len(colors_hot) - 1:
                color = colors_hot[color_idx] * (1 - color_t) + colors_hot[color_idx + 1] * color_t
            else:
                color = colors_hot[color_idx]

            opacity = 0.25 * (1 - t * 0.3)

            meshes.append({
                'mesh': sphere,
                'original_mesh': sphere.copy(),  # Store original
                'type': 'sphere',
                'color': color,
                'opacity': opacity,
                't': t,
                'base_z': z_pos,
                'base_radius': sphere_radius
            })

        # Create cylinders
        for i in range(self.n_cylinders):
            t = (i + 0.5) / self.n_cylinders
            z_pos = t * self.plume_length
            base_radius = self.size / 2
            expansion = 0.4 * t
            cyl_radius = base_radius + expansion
            cyl_height = self.plume_length / self.n_cylinders * 0.9

            cylinder = pv.Cylinder(
                center=(0, 0, z_pos),
                direction=(0, 0, 1),
                radius=cyl_radius,
                height=cyl_height,
                resolution=32
            )

            color_idx = min(int(t * (len(colors_hot) - 1)), len(colors_hot) - 1)
            color_t = (t * (len(colors_hot) - 1)) - color_idx

            if color_idx < len(colors_hot) - 1:
                color = colors_hot[color_idx] * (1 - color_t) + colors_hot[color_idx + 1] * color_t
            else:
                color = colors_hot[color_idx]

            opacity = 0.2 * (1 - t * 0.4)

            meshes.append({
                'mesh': cylinder,
                'original_mesh': cylinder.copy(),  # Store original
                'type': 'cylinder',
                'color': color,
                'opacity': opacity,
                't': t,
                'base_z': z_pos,
                'base_radius': cyl_radius,
                'height': cyl_height
            })

        return meshes

    @staticmethod
    def get_gimbal_transform(thrust_state: Union[np.ndarray, ca.MX]) -> Union[np.ndarray, ca.MX]:
        is_casadi = isinstance(thrust_state, (ca.SX, ca.MX))

        lib = ca if is_casadi else np

        # Select the correct trig functions and pi
        cos = lib.cos
        sin = lib.sin

        pi_val = lib.pi if is_casadi else np.pi

        # Extract angles
        gimbal_y_deg = thrust_state[1]
        gimbal_x_deg = thrust_state[2]

        # Convert to radians
        gimbal_x = gimbal_x_deg * (pi_val / 180)
        gimbal_y = gimbal_y_deg * (pi_val / 180)  # Negative for standard aerospace yaw

        cx, sx = cos(gimbal_x), sin(gimbal_x)
        cy, sy = cos(gimbal_y), sin(gimbal_y)

        # ca.blockcat is the CasADi equivalent of np.block
        if is_casadi:
            Rx = ca.blockcat([[1, 0, 0, 0],
                              [0, cx, -sx, 0],
                              [0, sx, cx, 0],
                              [0, 0, 0, 1]])

            Ry = ca.blockcat([[cy, 0, sy, 0],
                              [0, 1, 0, 0],
                              [-sy, 0, cy, 0],
                              [0, 0, 0, 1]])

        else:
            Rx = np.array([[1, 0, 0, 0],
                           [0, cx, -sx, 0],
                           [0, sx, cx, 0],
                           [0, 0, 0, 1]])

            Ry = np.array([[cy, 0, sy, 0],
                           [0, 1, 0, 0],
                           [-sy, 0, cy, 0],
                           [0, 0, 0, 1]])

        return Rx @ Ry

    def update_plume_transform(self, thrust_state: np.ndarray):
        """
        Update plume geometry by scaling z-position and applying gimbal rotation
        """
        #thrust_state = np.array([throttle, gimbal_x_deg, gimbal_y_deg])
        throttle = thrust_state[0]
        gimbal_rotation = self.get_gimbal_transform(thrust_state)

        for mesh_data in self.plume_meshes:
            t = mesh_data['t']

            # Only scale z position based on throttle
            z_offset = -t * self.plume_length * throttle

            # Create translation matrix
            translation = np.eye(4)
            translation[2, 3] = z_offset - mesh_data['base_z']

            # Apply transforms in order: translate -> reflect -> gimbal
            transformed = mesh_data['original_mesh'].copy()
            transformed.transform(translation)
            transformed.transform(gimbal_rotation)
            #transformed.transform(self.dynamic_transform_matrix)

            # Update the mesh points in place
            mesh_data['mesh'].points[:] = transformed.points

            # Update opacity
            new_opacity = mesh_data['opacity'] * throttle
            mesh_data['actor'].GetProperty().SetOpacity(new_opacity)
