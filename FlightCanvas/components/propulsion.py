
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
        self.thrust_direction = thrust_direction
        self.thrust_bounds = thrust_bounds
        self.gimbal_bounds = gimbal_bounds
        self.size = size
        self.x_direction = x_direction

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
        nozzle_exit = pv.Circle(radius=0.5, resolution=36)
        pl.add_mesh(nozzle_exit, color='black', line_width=3, style='wireframe')

        # Add rocket nozzle cylinder
        nozzle_cylinder = pv.Cylinder(
            center=self.xyz_ref,
            direction=(0, 0, 1),
            radius=0.6,
            height=2.0,
            resolution=32
        )
        nozzle_actor = pl.add_mesh(nozzle_cylinder, color='gray', smooth_shading=True)

    def init_debug(self, pl: pv.Plotter, com: np.ndarray, size: float = 1.0, label=True):
        default_sphere_radius = 0.02
        sphere_radius = default_sphere_radius * 2 ** (size - 1)

        # Draw a sphere at the component's reference point
        sphere = pv.Sphere(radius=sphere_radius, center=self.xyz_ref)
        ref_actor = pl.add_mesh(sphere, color='red', show_edges=False)

        # Draw an arrow from the component's ref point to the vehicle's CoM
        arrow_actor = utils.plot_line_from_points(pl, self.xyz_ref, com, color='grey')

    def update_actor(self, state: np.ndarray, thrust_state: np.ndarray):
        pass

    def get_transform(self, **kwargs) -> Union[np.ndarray, ca.MX]:
        pass

    def get_forces_and_moments(
            self,
            thrust: Union[float, ca.MX],
            gimbal_x: Union[float, ca.MX],
            gimbal_y: Union[float, ca.MX],
    ) -> Tuple[Union[np.ndarray, ca.MX], Union[np.ndarray, ca.MX]]:
        """
        TODO
        """
        pass


