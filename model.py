"""
model.py module contains the primary components for constructing the finite element model.

.. moduleauthor:: Tyler Ryan <tyler.ryan@engineering.ucla.edu>
"""
import numpy
from scipy.spatial import Delaunay

import constants
import elements
import nodes


class Model:
    """Finite Element Model containing globally needed values and functions for performing finite element analysis.

    :ivar material: material object that described the material the element is composed of
    :ivar constitutive_model: constitutive model class that describes the material behavior
    :ivar quadrature_class: quadrature class to use for elements
    :ivar element_type: element class used as a template for all finite elements
    :ivar list nodes: list of all node objects in the mesh
    :ivar list elements: list of element objects that make up the body
    """
    # TODO update docstring once finalized
    def __init__(self, material, constitutive_model, quadrature_class, element_type, degrees_of_freedom,
                 node_reference_positions_2d, node_reference_positions_3d, edges, corner_node_quantity,
                 prescribed_dof, membrane_thickness,
                 applied_load):
        # Inputs
        self.material = material
        self.constitutive_model = constitutive_model
        self.quadrature_class = quadrature_class
        self.element_type = element_type
        self.degrees_of_freedom = degrees_of_freedom
        self.node_reference_positions_2d = node_reference_positions_2d
        self.node_reference_positions_3d = node_reference_positions_3d
        self.edges = edges
        self.corner_node_quantity = corner_node_quantity
        self.prescribed_dof = prescribed_dof
        self.membrane_thickness = membrane_thickness
        self.applied_load = applied_load

        # Global quantities
        self.connectivity_table = None
        self.node_quantity = None
        self.nodes = []
        self.elements = []
        self.prescribed_dof_quantity = 0
        self.global_dof = 0

        # Updating quantities
        self.unknown_displacements = None
        self.strain_energy = None
        self.internal_force_array = None
        self.external_force_array = None
        self.stiffness_matrix = None

        # Run the analysis
        self.run()

    def assign_nodes(self):
        """Add nodes to their parent elements."""
        for node in self.nodes:
            for parent_element in node.parent_elements:
                parent_element.nodes.append(node)

    def create_connectivity_table(self):
        """Create connectivity table using Delaunay triangulation to connects nodes to elements."""
        delaunay_triangulation = Delaunay(self.node_reference_positions_2d)
        self.connectivity_table = delaunay_triangulation.simplices.copy()

    def create_corner_nodes(self):
        """Create corner nodes and add to the model."""
        for node_index in range(self.corner_node_quantity):
            node = nodes.CornerNode(global_id=node_index,
                                    reference_position=self.node_reference_positions_3d[node_index],
                                    prescribed_dof=self.prescribed_dof[node_index])
            self.nodes.append(node)

    def create_elements(self):
        """Create elements from the nodes using the connectivity table."""
        for element_index in range(len(self.connectivity_table)):
            element = self.element_type(constitutive_model=self.constitutive_model,
                                        material=self.material,
                                        quadrature_class=self.quadrature_class,
                                        degrees_of_freedom=self.degrees_of_freedom,
                                        thickness=self.membrane_thickness)
            # Add corner nodes to element
            for global_id in self.connectivity_table[element_index]:
                element.nodes.append(self.nodes[global_id])
            self.elements.append(element)

    def create_mesh(self):
        """Create the nodes, connectivity table, and elements for the model that make up the mesh."""
        self.create_corner_nodes()
        self.create_connectivity_table()
        self.create_elements()
        if self.element_type is elements.TriangularQuadraticElement:
            self.create_midpoint_nodes()

    def create_midpoint_nodes(self):
        """Create midpoint nodes for quadratic elements."""
        # Set global ID to corner node quantity so unique IDs can be assigned to new nodes
        global_id = self.corner_node_quantity
        existing_midpoint_nodes = {}
        for element in self.elements:
            node_pairs = [(element.nodes[0], element.nodes[1]),
                          (element.nodes[1], element.nodes[2]),
                          (element.nodes[2], element.nodes[0])]
            for node_pair in node_pairs:
                # Search for node pair in existing midpoint nodes
                for existing_pair in existing_midpoint_nodes:
                    # If a midpoint node has already been created for the node pair, use it
                    if set(node_pair) == set(existing_pair):
                        midpoint_node = existing_midpoint_nodes[existing_pair]
                        # Add the node to the element
                        element.nodes.append(midpoint_node)
                        # Once the existing node has been found, stop looking
                        break
                # Otherwise create a new midpoint node
                else:
                    # Position is the midpoint of the corner nodes
                    reference_position = .5 * (node_pair[0].reference_position + node_pair[1].reference_position)
                    # Set prescribed displacements based on corner nodes
                    prescribed_dof = [None] * self.degrees_of_freedom
                    # NOTE: if the midpoint is not on an edge, it cannot be prescribed in any way
                    for endpoints in self.edges:
                        cross_product = numpy.cross((endpoints[1] - endpoints[0]),
                                                    (reference_position[:2] - endpoints[0]))
                        # If the midpoint is on an edge, prescribe displacements
                        if abs(cross_product) < constants.FLOATING_POINT_TOLERANCE:
                            for dof_index in range(self.degrees_of_freedom):
                                displacement_1 = node_pair[0].prescribed_dof[dof_index]
                                displacement_2 = node_pair[1].prescribed_dof[dof_index]
                                # If there are prescribed displacements for both corner nodes
                                if displacement_1 is not None and displacement_2 is not None:
                                    # Prescribed displacement is the average of the corner nodes displacements
                                    prescribed_dof[dof_index] = .5 * (displacement_1 + displacement_2)
                    midpoint_node = nodes.MidpointNode(global_id=global_id,
                                                       reference_position=reference_position,
                                                       prescribed_dof=prescribed_dof)
                    # Add midpoint node to the model and element
                    self.nodes.append(midpoint_node)
                    element.nodes.append(midpoint_node)
                    # Add midpoint node to existing midpoint nodes
                    existing_midpoint_nodes[node_pair] = midpoint_node
                    # Increment the global ID
                    global_id += 1

    def run(self):
        """Run the analysis."""
        self.create_mesh()
        self.compute_node_and_dof_quantities()
        self.create_quadrature_points()

    def update_current_configuration(self):
        """Update the current configuration of all elements in the model."""
        for element in self.elements:
            element.update_current_configuration()

    def create_quadrature_points(self):
        """Create quadrature points for all elements."""
        for element in self.elements:
            element.create_quadrature_points()

    def compute_node_and_dof_quantities(self):
        """Compute the total number of nodes, the total number of global degrees of freedom, and the total number of
        prescribed degrees of freedom."""
        self.node_quantity = len(self.nodes)
        self.global_dof = self.node_quantity * self.degrees_of_freedom
        # Compute total number of prescribed degrees of freedom
        prescribed_dof_counter = 0
        for node in self.nodes:
            for dof in node.prescribed_dof:
                # If the dof is prescribed
                if dof is not None:
                    prescribed_dof_counter += 1
        self.prescribed_dof_quantity = prescribed_dof_counter

