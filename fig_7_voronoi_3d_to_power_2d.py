# -----------------------------------------------------------------------------
# Accompanying Code for:
# "SplInterp: Improving our Understanding and Training of Sparse Autoencoders" 
# (Anonymous, 2025)
# This script projects 3D Voronoi diagrams onto 2D power diagrams.
# The resulting power cells demonstrate the preservation of topological
# structure through projection.
#
# Please cite the above paper if you use this code in your work.
# -----------------------------------------------------------------------------

import numpy as np
import pyvista as pv
from scipy.spatial import distance_matrix
import matplotlib.cm as cm

np.random.seed(42)

# random 3D points
num_points = 15
points = np.random.rand(num_points, 3) * 2 - 1

# Create weights for the power diagram (from 3D z coordinates with some variation)
weights = 1.0 + 0.5 * (points[:, 2] + 1) + 0.3 * np.random.rand(num_points)
weights = 0.5 + 1.5 * (points[:, 2] + 1) + 0.5 * np.random.rand(num_points)
weights = 3.0 - 1.5 * (points[:, 2] + 1) - 0.5 * np.random.rand(num_points)

cloud = pv.PolyData(points)

# Create a 3D Voronoi diagram 
voronoi = cloud.delaunay_3d(alpha=1.0)
shell = voronoi.extract_geometry()

# Add cell scalars for multi-colored cells
shell['cell_id'] = np.arange(shell.n_cells)

# Create a plotter and add the elements
plotter = pv.Plotter(window_size=[1200, 900])

# Add the Voronoi cells with multi-color
plotter.add_mesh(
    shell, 
    scalars='cell_id',  
    cmap='rainbow',     
    show_scalar_bar=False,  
    opacity=0.4,        
    smooth_shading=True 
)

# Add points as spheres
plotter.add_points(
    points, 
    color='yellow',     
    point_size=7,      
    render_points_as_spheres=True  
)


# Position the projection plane below the structure
plane_z = -2.5
projection_plane = pv.Plane(
    center=(0, 0, plane_z), 
    direction=(0, 0, 1), 
    i_size=4, 
    j_size=4
)

# Create a grid on the plane for power diagram
grid = pv.ImageData(
    dimensions=(200, 200, 1),  
    origin=(-2, -2, plane_z),
    spacing=(0.02, 0.02, 0.05)  
)

# Extract just points within a 2D circle for cleaner projection
base_points = points[:, 0:2]
points_2d = grid.points[:, :2]

# Compute power distances for the projection
def power_distance(x, centers, weights):
    return np.sum((x[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2) - weights[np.newaxis, :]

distances = power_distance(points_2d, base_points, weights)
cell_assignments = np.argmin(distances, axis=1)

# Create power diagram
power_diagram = grid.copy()
power_diagram["cell_id"] = cell_assignments

# Add projection plane with power diagram
plotter.add_mesh(
    power_diagram,
    scalars='cell_id',
    cmap='viridis',
    show_scalar_bar=False,
    opacity=0.7,
)

# plane frame for clarity
plane_edges = projection_plane.extract_feature_edges()
plotter.add_mesh(plane_edges, color='black', line_width=2)

for i in range(len(points)):
    # Create line from 3D point to projection
    point_3d = points[i]
    projection_point = np.array([point_3d[0], point_3d[1], plane_z])
    line = pv.Line(point_3d, projection_point)
    plotter.add_mesh(line, color='purple', line_width=1.5, opacity=0.7)
    
    # Add marker on the projection point
    sphere = pv.Sphere(radius=0.03, center=projection_point)
    plotter.add_mesh(sphere, color='black', opacity=0.9)
    
    # Wireframe circle on the projection plane
    radius = np.sqrt(weights[i]) * 0.2  # Scale for better visualization
    
    # polyline approach instead of Circle object for better wireframe rendering
    theta = np.linspace(0, 2*np.pi, 60)  
    circle_points = np.zeros((60, 3))
    circle_points[:, 0] = point_3d[0] + radius * np.cos(theta)
    circle_points[:, 1] = point_3d[1] + radius * np.sin(theta)
    circle_points[:, 2] = plane_z + 0.01  # Slightly above plane
        
    # polyline for the circle
    circle_poly = pv.PolyData(circle_points)
    circle_poly.lines = np.hstack([[60], np.arange(60), [0]])  # Close the loop
    plotter.add_mesh(circle_poly, color='white', line_width=1.5, opacity=0.5) 
    
    if i % 3 == 0:  # Only show labels for some points to avoid clutter
        plotter.add_point_labels(
            [point_3d + np.array([0, 0, 0.15])],
            [f"w={weights[i]:.1f}"],
            font_size=22,
            text_color='black',
            shape_color='#e0e0e0',
            always_visible=True,
            shadow=True,
            shape='rounded_rect',
            fill_shape=True
        )
    
    # Add some radial vectors for select points
    if i % 3 == 0:  # Show vectors for select points only
        for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
            end_point = np.array([
                point_3d[0] + radius * np.cos(angle),
                point_3d[1] + radius * np.sin(angle),
                plane_z + 0.01
            ])
            vector_line = pv.Line(projection_point, end_point)
            plotter.add_mesh(vector_line, color='white', line_width=1.5)

plotter.add_bounding_box(color='gray', opacity=0.3)
plotter.show_grid(
    xlabel='',
    ylabel='',
    zlabel='',
    ticks=None,
    xtitle='',
    ytitle='',
    ztitle='',
    color='#bbbbbb',  
    )

plotter.add_text(
    "3D Voronoi Projecting to 2D Power Diagram",
    position='upper_edge',
    font_size=17,
    color='black',
    shadow=True
)

plotter.add_text(
    "Higher centers → Lower weights → Smaller power regions below",
    position='lower_edge',
    font_size=12,
    color='black',
    shadow=True
)

# Set background and camera
plotter.set_background('#eeeeee', top='#ffffff')
plotter.view_isometric()
plotter.camera_position = [7, 7, 5]  
plotter.camera.zoom(0.9)

plotter.show()

