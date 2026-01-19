import pyvista as pv
import numpy as np


def create_thrust_plume(base_radius, height, tip_radius):
    """
    Create a truncated cone (cylinder with wider bottom) thrust plume.

    Parameters:
    base_radius: float - radius at the base (top, narrower end at nozzle)
    height: float - height of the plume
    tip_radius: float - radius at the tip (bottom, wider end)

    Returns:
    pv.PolyData: The thrust plume mesh
    """
    # Create points for the truncated cone
    resolution = 50
    theta = np.linspace(0, 2 * np.pi, resolution)

    # Top circle (at nozzle, smaller radius)
    top_x = base_radius * np.cos(theta)
    top_y = np.zeros(resolution)
    top_z = base_radius * np.sin(theta)

    # Bottom circle (wider)
    bottom_x = tip_radius * np.cos(theta)
    bottom_y = -height * np.ones(resolution)
    bottom_z = tip_radius * np.sin(theta)

    # Combine points
    points = np.column_stack([
        np.concatenate([top_x, bottom_x]),
        np.concatenate([top_y, bottom_y]),
        np.concatenate([top_z, bottom_z])
    ])

    # Create faces connecting top and bottom circles
    faces = []
    for i in range(resolution):
        next_i = (i + 1) % resolution
        # Create two triangles for each quad
        faces.append([3, i, next_i, resolution + i])
        faces.append([3, next_i, resolution + next_i, resolution + i])

    faces = np.hstack(faces)

    # Create mesh
    plume = pv.PolyData(points, faces)

    return plume


# Define plume parameters
base_radius = 1.0  # Radius at the top (nozzle, narrower)
height = 20.0  # Height of plume
tip_radius = 2.0  # Radius at the bottom (wider)

# Create plotter
pl = pv.Plotter()

# Create and add the thrust plume
thrust = create_thrust_plume(base_radius, height, tip_radius)
pl.add_mesh(thrust, color='orange', opacity=0.5, smooth_shading=True)

# Configure view
pl.camera_position = [(10, 0, 0), (0, 0, 0), (0, 1, 0)]
pl.add_text(f'Base: {base_radius}, Height: {height}, Tip: {tip_radius}',
            position='upper_left', font_size=12)

# Show plot
pl.show()