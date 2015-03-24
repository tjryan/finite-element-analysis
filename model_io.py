"""
model_io.py contains the interface to the model.
"""
import itertools

import numpy
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

import constitutive_models
import elements
import quadrature
import materials
import model



# INPUTS
material = materials.Custom('custom', 4e6, 4e5)
constitutive_model = constitutive_models.Neohookean
quadrature_class = quadrature.GaussQuadratureOnePoint
# quadrature_class = quadrature.GaussQuadratureThreePoint
element_type = elements.TriangularLinearElement
# element_type = elements.TriangularQuadraticElement
degrees_of_freedom = 3
membrane_thickness = .001

### SQUARE MESH
# Inputs for creating square mesh
side_length = .1
nodes_per_side = 3
stretch_percent = 0.0

# Determine nodal positions
x_coordinates = numpy.linspace(0, side_length, nodes_per_side)
y_coordinates = numpy.linspace(0, side_length, nodes_per_side)
node_reference_positions_2d = []
for xy_point in itertools.product(x_coordinates, y_coordinates):
    node_reference_positions_2d.append(xy_point)
node_reference_positions_2d = numpy.array(node_reference_positions_2d, dtype=float)
corner_node_quantity = node_reference_positions_2d.shape[0]

# Specify sets of edge endpoints
edge_1 = [(0, 0), (side_length, 0)]
edge_2 = [(side_length, 0), (side_length, side_length)]
edge_3 = [(side_length, side_length), (0, side_length)]
edge_4 = [(0, side_length), (0, 0)]
edges = numpy.array([edge_1, edge_2, edge_3, edge_4], dtype=float)

# Initialize prescribed displacements, by default, nothing is prescribed
prescribed_displacements = {}
for node_index in range(corner_node_quantity):
    prescribed_displacements[node_index] = [None, None, None]
# Set prescribed displacements
for point_index in range(corner_node_quantity):
    position = node_reference_positions_2d[point_index]
    # If the node is along any edge
    if position[0] == 0 or position[0] == side_length or position[1] == 0 or position[1] == side_length:
        prescribed_displacements[point_index] = [stretch_percent * position[0], stretch_percent * position[1], 0]


# Make single element for testing
# node_reference_positions_2d = numpy.array([[0, 0], [1, 0], [0, 1]], dtype=float)
# corner_node_quantity = node_reference_positions_2d.shape[0]
# edges = numpy.array([[(0, 0), (1, 0)], [(0, 0), (0, 1)], [(0, 1), (0, 1)]], dtype=float)

# Display mesh
delaunay_triangulation = Delaunay(node_reference_positions_2d)
plt.triplot(node_reference_positions_2d[:, 0], node_reference_positions_2d[:, 1],
            delaunay_triangulation.simplices.copy())
plt.plot(node_reference_positions_2d[:, 0], node_reference_positions_2d[:, 1], 'o')
plt.title('2D Mesh (Coarse)')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.show()

# Convert 2D nodal positions to 3D (this may change depending on body being modelled)
node_reference_positions_3d = []
# Flat sheet
for position in node_reference_positions_2d:
    node_reference_positions_3d.append(list(numpy.append(position, 0)))
node_reference_positions_3d = numpy.array(node_reference_positions_3d, dtype=float)

# Set applied loading
applied_load = numpy.array([0, 0, -100], dtype=float)

# Set number of load/displacement steps
step_quantity = 10

# Set problem type
solve_loading_problem = False
solve_displacement_problem = True


### Octant of sphere
# # Inputs
# radius = 10
# circle_quantity = 6
#
# # Calculate 2d positions
# node_positions_2d = []
# node_positions_3d = []
# radii = numpy.linspace(0, radius, circle_quantity)
# for circle_index in range(circle_quantity):
# circle_radius = radii[circle_index]
# # Calculate angles to sweep over
# angles = numpy.linspace(0, 90, circle_index + 1)
#     for angle in angles:
#         x_position = circle_radius * numpy.cos(angle * numpy.pi / 180)
#         if abs(x_position) < 1e-6:
#             x_position = 0
#         y_position = circle_radius * numpy.sin(angle * numpy.pi / 180)
#         if abs(y_position) < 1e-6:
#             y_position = 0
#         z_position = numpy.sqrt(abs(radius ** 2 - x_position ** 2 - y_position ** 2))
#         if abs(z_position) < 1e-6:
#             z_position = 0
#         node_positions_2d.append((x_position, y_position))
#         node_positions_3d.append((x_position, y_position, z_position))
# node_reference_positions_2d = numpy.array(node_positions_2d)
# corner_node_quantity = node_reference_positions_2d.shape[0]
# node_reference_positions_3d = numpy.array(node_positions_3d)
# # Display mesh
# delaunay_triangulation = Delaunay(node_reference_positions_2d)
# plt.triplot(node_reference_positions_2d[:, 0], node_reference_positions_2d[:, 1],
#             delaunay_triangulation.simplices.copy())
# plt.plot(node_reference_positions_2d[:, 0], node_reference_positions_2d[:, 1], 'o')
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_trisurf(node_reference_positions_3d[:, 0], node_reference_positions_3d[:, 1], node_reference_positions_3d[:, 2],
#                 triangles=delaunay_triangulation.simplices.copy(), alpha=.5)
# plt.show()
#
# # Initialize prescribed displacements, by default, nothing is prescribed
# prescribed_displacements = {}
# for node_index in range(corner_node_quantity):
#     prescribed_displacements[node_index] = [None, None, None]
# for node_index in range(corner_node_quantity):
#     for dof in range(3):
#         if node_reference_positions_3d[node_index][dof] == 0:
#             prescribed_displacements[node_index][dof] = 0

# Create Model and run
model = model.Model(material=material,
                    constitutive_model=constitutive_model,
                    quadrature_class=quadrature_class,
                    element_type=element_type,
                    degrees_of_freedom=degrees_of_freedom,
                    node_reference_positions_2d=node_reference_positions_2d,
                    node_reference_positions_3d=node_reference_positions_3d,
                    edges=edges,
                    corner_node_quantity=corner_node_quantity,
                    prescribed_displacements=prescribed_displacements,
                    membrane_side_length=side_length,
                    membrane_thickness=membrane_thickness,
                    applied_load=applied_load,
                    step_quantity=step_quantity,
                    solve_loading_problem=solve_loading_problem,
                    solve_displacement_problem=solve_displacement_problem)
