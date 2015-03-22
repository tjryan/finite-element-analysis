"""
model_io.py contains the interface to the model.
"""
import numpy
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

import constitutive_models
import elements
import quadrature
import materials
import model











# INPUTS
material = materials.Custom('custom', 10, 5)
constitutive_model = constitutive_models.Neohookean
quadrature_class = quadrature.GaussQuadratureOnePoint
# quadrature_class = quadrature.GaussQuadratureThreePoint
element_type = elements.TriangularLinearElement
# element_type = elements.TriangularQuadraticElement
degrees_of_freedom = 3
membrane_thickness = 1

# Set 2D node reference positions
node_reference_positions_2d = numpy.array([[0, 0], [1, 0], [2, 0],
                                           [0, 1], [1, 1], [2, 1],
                                           [0, 2], [1, 2], [2, 2]], dtype=float)
corner_node_quantity = node_reference_positions_2d.shape[0]
# Specify sets of edge endpoints
edges = numpy.array([[(0, 0), (2, 0)], [(0, 0), (0, 2)], [(2, 0), (2, 2)], [(0, 2), (2, 2)]], dtype=float)

# Display mesh
delaunay_triangulation = Delaunay(node_reference_positions_2d)
plt.triplot(node_reference_positions_2d[:, 0], node_reference_positions_2d[:, 1],
            delaunay_triangulation.simplices.copy())
plt.plot(node_reference_positions_2d[:, 0], node_reference_positions_2d[:, 1], 'o')
# plt.show()

# Convert 2D nodal positions to 3D (this may change depending on body being modelled)
node_reference_positions_3d = []
# Flat sheet
for position in node_reference_positions_2d:
    node_reference_positions_3d.append(list(numpy.append(position, 0)))
node_reference_positions_3d = numpy.array(node_reference_positions_3d, dtype=float)

# Initialize prescribed displacements, be default, nothing is prescribed
prescribed_dof = {}
for node_index in range(corner_node_quantity):
    prescribed_dof[node_index] = [None, None, None]

# Set nodes and prescribed displacements
prescribed_dof[0] = [0, 0, 0]
prescribed_dof[1] = [0, 0, 0]
prescribed_dof[2] = [0, 0, 0]
prescribed_dof[3] = [0, 0, 0]
prescribed_dof[5] = [0, 0, 0]
prescribed_dof[6] = [0, 0, 0]
prescribed_dof[7] = [0, 0, 0]
prescribed_dof[8] = [0, 0, 0]
prescribed_dof[9] = [0, 0, 0]

# Set applied loading
applied_load = 1000

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
                    prescribed_dof=prescribed_dof,
                    membrane_thickness=membrane_thickness,
                    applied_load=applied_load)
