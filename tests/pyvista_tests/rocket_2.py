import pyvista as pv
import numpy as np

# Create plotter
pl = pv.Plotter()
pl.background_color = 'white'

# Storage for plume meshes and actors
plume_meshes = []

# Color gradient from hot (yellow) to cool (red/dark)
colors_hot = np.array([
    [1.0, 1.0, 0.6],  # Light yellow
    [1.0, 0.9, 0.3],  # Yellow
    [1.0, 0.7, 0.2],  # Orange-yellow
    [1.0, 0.5, 0.1],  # Orange
    [1.0, 0.3, 0.0],  # Red-orange
    [0.8, 0.2, 0.0],  # Dark red
    [0.6, 0.1, 0.0],  # Very dark red
])


# Function to create all plume geometry once
def create_plume_geometry():
    """Create all plume meshes at full throttle"""
    plume_length = 10.0
    n_cones = 7
    n_spheres = 0
    n_cylinders = 0

    meshes = []

    # Create cones
    for i in range(n_cones):
        t = i / n_cones
        z_pos = -t * plume_length
        base_radius = 0.6
        expansion = 0.4 * t
        radius = base_radius + expansion
        cone_height = 10 / n_cones * 1.2

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
    for i in range(n_spheres):
        t = i / n_spheres
        z_pos = -t * plume_length
        base_radius = 0.6
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
    for i in range(n_cylinders):
        t = (i + 0.5) / n_cylinders
        z_pos = -t * plume_length
        base_radius = 0.6
        expansion = 0.4 * t
        cyl_radius = base_radius + expansion
        cyl_height = plume_length / n_cylinders * 0.9

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


# Create all meshes once
plume_meshes = create_plume_geometry()

# Add all meshes to plotter
for mesh_data in plume_meshes:
    mesh_data['actor'] = pl.add_mesh(
        mesh_data['mesh'],
        color=mesh_data['color'],
        opacity=mesh_data['opacity'],
        smooth_shading=True
    )

# Add nozzle exit plane for reference
nozzle_exit = pv.Circle(radius=0.5, resolution=36)
pl.add_mesh(nozzle_exit, color='black', line_width=3, style='wireframe')

# Add text display for throttle and gimbal
throttle = 1.0
gimbal_x = 0.0  # Gimbal angle in degrees (pitch)
gimbal_y = 0.0  # Gimbal angle in degrees (yaw)
text_actor = pl.add_text(f'Throttle: {throttle * 100:.0f}%\nGimbal X: {gimbal_x:.1f}°\nGimbal Y: {gimbal_y:.1f}°',
                         position='upper_left',
                         font_size=12,
                         color='black')


# Function to update plume based on throttle
def update_plume_transform(throttle, gimbal_x_deg=0.0, gimbal_y_deg=0.0):
    """Update plume geometry by scaling z-position and applying gimbal rotation"""
    plume_length = 10.0

    # Convert gimbal angles to radians
    gimbal_x = np.radians(gimbal_x_deg)
    gimbal_y = np.radians(gimbal_y_deg)

    # Create rotation matrix for gimbal
    # Rotation around X axis (pitch)
    Rx = np.array([
        [1, 0, 0, 0],
        [0, np.cos(gimbal_x), -np.sin(gimbal_x), 0],
        [0, np.sin(gimbal_x), np.cos(gimbal_x), 0],
        [0, 0, 0, 1]
    ])

    # Rotation around Y axis (yaw)
    Ry = np.array([
        [np.cos(gimbal_y), 0, np.sin(gimbal_y), 0],
        [0, 1, 0, 0],
        [-np.sin(gimbal_y), 0, np.cos(gimbal_y), 0],
        [0, 0, 0, 1]
    ])

    # Combine rotations (apply Y then X)
    gimbal_rotation = Rx @ Ry

    for mesh_data in plume_meshes:
        t = mesh_data['t']

        # Only scale z position based on throttle
        z_offset = -t * plume_length * throttle

        # Create translation matrix
        translation = np.eye(4)
        translation[2, 3] = z_offset - mesh_data['base_z']

        # Apply transform to ORIGINAL mesh: first translate, then rotate around origin
        transformed = mesh_data['original_mesh'].copy()
        transformed.transform(translation)
        transformed.transform(gimbal_rotation)

        # Update the mesh points in place
        mesh_data['mesh'].points[:] = transformed.points

        # Update opacity
        new_opacity = mesh_data['opacity'] * throttle
        mesh_data['actor'].GetProperty().SetOpacity(new_opacity)


# Callback for throttle control
def update_throttle(value):
    global throttle
    throttle = value / 100.0

    # Update plume transformation
    update_plume_transform(throttle, gimbal_x, gimbal_y)

    # Update text
    text_actor.SetText(2, f'Throttle: {throttle * 100:.0f}%\nGimbal X: {gimbal_x:.1f}°\nGimbal Y: {gimbal_y:.1f}°')


# Callback for gimbal X control
def update_gimbal_x(value):
    global gimbal_x
    gimbal_x = value

    # Update plume transformation
    update_plume_transform(throttle, gimbal_x, gimbal_y)

    # Update text
    text_actor.SetText(2, f'Throttle: {throttle * 100:.0f}%\nGimbal X: {gimbal_x:.1f}°\nGimbal Y: {gimbal_y:.1f}°')


# Callback for gimbal Y control
def update_gimbal_y(value):
    global gimbal_y
    gimbal_y = value

    # Update plume transformation
    update_plume_transform(throttle, gimbal_x, gimbal_y)

    # Update text
    text_actor.SetText(2, f'Throttle: {throttle * 100:.0f}%\nGimbal X: {gimbal_x:.1f}°\nGimbal Y: {gimbal_y:.1f}°')


# Add slider for throttle control
pl.add_slider_widget(
    update_throttle,
    [0, 100],
    value=100,
    title="Throttle %",
    pointa=(0.1, 0.1),
    pointb=(0.4, 0.1),
    style='modern'
)

# Add slider for gimbal X (pitch)
pl.add_slider_widget(
    update_gimbal_x,
    [-15, 15],
    value=0,
    title="Gimbal X (°)",
    pointa=(0.1, 0.18),
    pointb=(0.4, 0.18),
    style='modern'
)

# Add slider for gimbal Y (yaw)
pl.add_slider_widget(
    update_gimbal_y,
    [-15, 15],
    value=0,
    title="Gimbal Y (°)",
    pointa=(0.1, 0.26),
    pointb=(0.4, 0.26),
    style='modern'
)

# Add coordinate axes for reference
pl.add_axes(xlabel='X', ylabel='Y', zlabel='Z')

# Set camera position for good view
pl.camera_position = [(6, 6, 3), (0, 0, -3), (0, 0, 1)]

# Enable anti-aliasing for smooth rendering
pl.enable_anti_aliasing()

# Show plot
pl.show()