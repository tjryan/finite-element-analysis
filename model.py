"""
model.py module contains the primary components for constructing the finite element model.

.. moduleauthor:: Tyler Ryan <tyler.ryan@engineering.ucla.edu>
"""

import numpy
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

import constants
import elements
import nodes


class Model:
    """Finite Element Model that sets up and runs the finite element analysis for the given problem.

    :param material: material object that described the material the element is composed of
    :param constitutive_model: constitutive model class that describes the material behavior
    :param quadrature_class: quadrature class to use for elements
    :param element_type: element class used as a template for all finite elements
    :param degrees_of_freedom: number of degrees of freedom at each node of the model (or number of dimensions of lab
    frame
    :param numpy.ndarray node_reference_positions_2d: array of 2D node positions in flat plane
    :param numpy.ndarray node_reference_positions_3d: array of node positions projected onto body in 3D
    :param numpy.ndarray edges: array of tuples containing 2D positions of edge endpoints
    :param int corner_node_quantity = number of corner nodes in the mesh
    :param dict prescribed_displacements = dictionary keying node ID to a list of prescribed degrees of freedom (for
    every corner node)
    :param float membrane_side_length: side length of square membrane
    :param float membrane_thickness: thickness of the membrane
    :param numpy.ndarray applied_load: vector of uniform transverse load applied to the membrane (force/area)
    """

    def __init__(self, material, constitutive_model, quadrature_class, element_type, degrees_of_freedom,
                 node_reference_positions_2d, node_reference_positions_3d, edges, corner_node_quantity,
                 prescribed_displacements, membrane_side_length, membrane_thickness, applied_load, step_quantity,
                 solve_loading_problem=False,
                 solve_displacement_problem=False):
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
        self.prescribed_displacements = prescribed_displacements
        self.membrane_side_length = membrane_side_length
        self.membrane_thickness = membrane_thickness
        self.applied_load = applied_load
        self.step_quantity = step_quantity
        self.solve_loading_problem = solve_loading_problem
        self.solve_displacement_problem = solve_displacement_problem

        # Global quantities
        self.connectivity_table = None
        self.node_quantity = None
        self.nodes = []
        self.element_quantity = 0
        self.elements = []
        self.global_dof_quantity = 0
        self.load_step = applied_load / step_quantity
        self.known_displacements = None
        self.known_displacement_quantity = 0
        self.unknown_displacement_quantity = 0

        # Updating quantities
        self.unknown_displacements = None
        self.strain_energy = None
        self.internal_force_array = None
        self.external_force_array = None
        self.stiffness_matrix = None

        # Outputs
        self.load_steps = []
        self.maximum_deflections = []

        # Run the analysis
        self.run()

    def calculate_node_and_dof_quantities(self):
        """Compute the total number of nodes, the total number of global degrees of freedom, and the total number of
        prescribed degrees of freedom."""
        self.node_quantity = len(self.nodes)
        self.element_quantity = len(self.elements)
        self.global_dof_quantity = self.node_quantity * self.degrees_of_freedom
        # Create array of known displacements
        known_displacements = []
        for node_id in self.prescribed_displacements:
            for dof in self.prescribed_displacements[node_id]:
                if dof is not None:
                    known_displacements.append(dof)
        self.known_displacements = numpy.array(known_displacements, dtype=float)
        self.known_displacement_quantity = self.known_displacements.size
        self.unknown_displacement_quantity = self.global_dof_quantity - self.known_displacement_quantity
        self.unknown_displacements = numpy.array([0] * self.unknown_displacement_quantity)

    @staticmethod
    def calculate_residual(external_force_array, internal_force_array):
        """Calculate the residual as the difference between the external and internal force arrays for the "upper"
        part only.

        :param external_force_array: external_force array for unknown degrees of freedom
        :param internal_force_array: internal_force array for unknown degrees of freedom
        """
        residual = (external_force_array - internal_force_array)
        return residual

    def create_connectivity_table(self):
        """Create connectivity table using Delaunay triangulation to connects nodes to elements."""
        delaunay_triangulation = Delaunay(self.node_reference_positions_2d)
        simplices = delaunay_triangulation.simplices.copy()
        # Sort simplices by node ID
        for node_set in simplices:
            node_set.sort()
        self.connectivity_table = simplices

    def create_corner_nodes(self):
        """Create corner nodes and add to the model."""
        for node_index in range(self.corner_node_quantity):
            node = nodes.CornerNode(global_id=node_index,
                                    reference_position=self.node_reference_positions_3d[node_index],
                                    prescribed_displacements=self.prescribed_displacements[node_index])
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
                    prescribed_displacements = [None] * self.degrees_of_freedom
                    # NOTE: if the midpoint is not on an edge, it cannot be prescribed in any way
                    for endpoints in self.edges:
                        cross_product = numpy.cross((endpoints[1] - endpoints[0]),
                                                    (reference_position[:2] - endpoints[0]))
                        # If the midpoint is on an edge, prescribe displacements
                        if abs(cross_product) < constants.FLOATING_POINT_TOLERANCE:
                            for dof_index in range(self.degrees_of_freedom):
                                displacement_1 = node_pair[0].prescribed_displacements[dof_index]
                                displacement_2 = node_pair[1].prescribed_displacements[dof_index]
                                # If there are prescribed displacements for both corner nodes
                                if displacement_1 is not None and displacement_2 is not None:
                                    # Prescribed displacement is the average of the corner nodes displacements
                                    prescribed_displacements[dof_index] = .5 * (displacement_1 + displacement_2)
                    midpoint_node = nodes.MidpointNode(global_id=global_id,
                                                       reference_position=reference_position,
                                                       prescribed_displacements=prescribed_displacements)
                    # Add midpoint node to the model and element
                    self.nodes.append(midpoint_node)
                    element.nodes.append(midpoint_node)
                    # Add midpoint node to existing midpoint nodes
                    existing_midpoint_nodes[node_pair] = midpoint_node
                    # Add midpoint node to prescribed displacements
                    self.prescribed_displacements[midpoint_node.global_id] = prescribed_displacements
                    # Increment the global ID
                    global_id += 1

    def create_quadrature_points(self):
        """Create quadrature points for all elements."""
        for element in self.elements:
            element.create_quadrature_points()

    def displacement_solver(self):
        """Solve for the deformation of the body based on the applied prescribed displacements. Uses the Newton-Raphson
        method to increment the deformation and iteratively solve the unknown displacements at each node for each step.
        """
        # Set displacement step for all nodes with prescribed degrees of freedom
        for node in self.nodes:
            for dof_index in range(self.degrees_of_freedom):
                # If there is a prescribed displacement, set the load step and displace the node
                if node.prescribed_displacements[dof_index] is not None:
                    node.displacement_step[dof_index] = node.prescribed_displacements[dof_index] / self.step_quantity
                    node.current_position[dof_index] += node.displacement_step[dof_index]
        # Increment displacements up to total prescribed displacements
        for displacement_step_index in range(self.step_quantity):
            print('Progress:', displacement_step_index / self.step_quantity * 100, '%')
            # Displace nodes for this step
            for node in self.nodes:
                for dof_index in range(self.degrees_of_freedom):
                    # If there is a prescribed displacement, set the load step and displace the node
                    if node.prescribed_displacements[dof_index] is not None:
                        node.current_position[dof_index] += node.displacement_step[dof_index]
            # Update the configuration
            self.update_current_configuration()
            self.update_plot()
            # Compute the external force array, which will be zero
            self.global_external_force_array(current_load=numpy.array([0, 0, 0]))
            # Initialize residual to be large
            residual = numpy.array([float('inf')] * self.degrees_of_freedom, dtype=float)
            # Loop until the residual is within tolerance of 0
            while abs(residual.flat[abs(residual).argmax()]) > constants.NEWTON_METHOD_TOLERANCE:
                # Rearrange global internal force and stiffness matrix to move prescribe degrees of freedom to the end
                self.rearrange_global()
                # Only work with "upper" equations to calculate the residual
                upper_external_force_array = self.external_force_array[:self.unknown_displacement_quantity]
                upper_internal_force_array = self.internal_force_array[:self.unknown_displacement_quantity]
                upper_stiffness_matrix = self.stiffness_matrix[:self.unknown_displacement_quantity,
                                         :self.unknown_displacement_quantity]
                residual = self.calculate_residual(external_force_array=upper_external_force_array,
                                                   internal_force_array=upper_internal_force_array)
                # Solve for the unknown displacements u = K^-1*(residual)
                stiffness_matrix_inverse = numpy.linalg.inv(upper_stiffness_matrix)
                self.unknown_displacements = numpy.dot(stiffness_matrix_inverse, residual)
                # Update model configuration for the new displacements
                self.update_current_configuration()
            # Update the membrane plot
            self.update_plot()

    def output_results(self):
        """Provide output data at end of analysis."""
        # Plot maximum deflection vs. load step
        plt.figure(2)
        plt.plot(self.load_steps, self.maximum_deflections)
        plt.title('Deflection')
        plt.xlabel('Load Magnitude (N/m^2)')
        plt.ylabel('Max deflection (m)')
        plt.show()

    def global_external_force_array(self, current_load):
        """Update the elements, then assemble and unroll the global external force array from the current load.

        :param current_load: current magnitude of the applied transverse load
        """
        # Update the external force array for the elements
        for element in self.elements:
            element.update_external_force_array(current_load)
        # Assemble global external force array
        dimensions = (self.degrees_of_freedom, self.node_quantity)
        external_force_array = numpy.zeros(dimensions)
        for element in self.elements:
            for dof in range(self.degrees_of_freedom):
                for node in element.nodes:
                    external_force_array[dof][node.global_id] += (
                        element.external_force_array[dof][element.nodes.index(node)]
                    )
        # Unroll the array into a column vector
        unrolled_dimensions = self.degrees_of_freedom * self.node_quantity
        external_force_array_unrolled = numpy.zeros(unrolled_dimensions)
        for dof in range(self.degrees_of_freedom):
            for node_index in range(self.node_quantity):
                index = self.degrees_of_freedom * node_index + dof
                external_force_array_unrolled[index] = external_force_array[dof][node_index]
        self.external_force_array = external_force_array_unrolled

    def global_internal_force_array(self):
        """Assemble and unroll the global internal force array by adding the contributions of the elements."""
        dimensions = (self.degrees_of_freedom, self.node_quantity)
        internal_force_array = numpy.zeros(dimensions)
        for element in self.elements:
            for dof in range(self.degrees_of_freedom):
                for node in element.nodes:
                    internal_force_array[dof][node.global_id] += (
                        element.internal_force_array[dof][element.nodes.index(node)]
                    )
        # Unroll the array into a column vector
        unrolled_dimensions = self.degrees_of_freedom * self.node_quantity
        internal_force_array_unrolled = numpy.zeros(unrolled_dimensions)
        for dof in range(self.degrees_of_freedom):
            for node_index in range(self.node_quantity):
                index = self.degrees_of_freedom * node_index + dof
                internal_force_array_unrolled[index] = internal_force_array[dof][node_index]
        self.internal_force_array = internal_force_array_unrolled

    def global_stiffness_matrix(self):
        """Assemble and unroll the global stiffness by adding the contributions of the elements."""
        dimensions = (self.degrees_of_freedom, self.node_quantity, self.degrees_of_freedom, self.node_quantity)
        stiffness_matrix = numpy.zeros(dimensions)
        for element in self.elements:
            for dof_1 in range(self.degrees_of_freedom):
                for node_1 in element.nodes:
                    for dof_2 in range(self.degrees_of_freedom):
                        for node_2 in element.nodes:
                            stiffness_matrix[dof_1][node_1.global_id][dof_2][node_2.global_id] += (
                                element.stiffness_matrix[dof_1][element.nodes.index(node_1)][dof_2][
                                    element.nodes.index(node_2)]
                            )
        # Unroll the 4th order tensor into a 2D matrix
        unrolled_dimensions = (self.degrees_of_freedom * self.node_quantity,
                               self.degrees_of_freedom * self.node_quantity)
        stiffness_matrix_unrolled = numpy.zeros(unrolled_dimensions)
        for dof_1 in range(self.degrees_of_freedom):
            for node_index_1 in range(self.node_quantity):
                for dof_2 in range(self.degrees_of_freedom):
                    for node_index_2 in range(self.node_quantity):
                        index_1 = self.degrees_of_freedom * node_index_1 + dof_1
                        index_2 = self.degrees_of_freedom * node_index_2 + dof_2
                        stiffness_matrix_unrolled[index_1][index_2] = (
                            stiffness_matrix[dof_1][node_index_1][dof_2][node_index_2]
                        )
        self.stiffness_matrix = stiffness_matrix_unrolled

    def global_strain_energy(self):
        """Calculate the global strain energy by adding the strain energies of the elements."""
        strain_energy = 0
        for element in self.elements:
            strain_energy += element.strain_energy
        self.strain_energy = strain_energy

    def loading_solver(self):
        """Solve for the deformation of the body based on the applied loading. Uses the Newton-Raphson method to
        increment the external loading and iteratively solve the unknown displacements at each node for each step.
        """
        # Initialize the load as a zero vector
        current_load = numpy.array([0] * self.degrees_of_freedom, dtype=float)
        # Initialize small random displacements for the unknown degrees of freedom
        # Perturb unconstrained nodes in the 3 direction
        for node in self.nodes:
            if node.prescribed_displacements[2] is None:
                node.current_position[2] += -1e-3 * (
                    numpy.sin(numpy.pi * node.current_position[0] / self.membrane_side_length)
                    * numpy.sin(numpy.pi * node.current_position[1] / self.membrane_side_length))
        # Update the configuration of the model to calculate the global strain energy, internal force, and stiffness
        # associated with the small random displacements
        self.update_current_configuration()
        self.update_plot()
        # Increment load up to total applied load
        for load_step_index in range(self.step_quantity):
            print('Progress:', load_step_index / self.step_quantity * 100, '%')
            # Increment the current load
            current_load += self.load_step
            # Calculate the global external force array for the current load
            self.global_external_force_array(current_load)
            # Rearrange global external force array to move prescribed degrees of freedom to the end
            self.rearrange_global_external_force_array()
            # Initialize residual to be large
            residual = numpy.array([float('inf')] * self.degrees_of_freedom, dtype=float)
            # Loop until the residual is within tolerance of 0
            while abs(residual.flat[abs(residual).argmax()]) > constants.NEWTON_METHOD_TOLERANCE:
                # Rearrange global internal force and stiffness matrix to move prescribe degrees of freedom to the end
                self.rearrange_global()
                # Only work with "upper" equations to calculate the residual
                upper_external_force_array = self.external_force_array[:self.unknown_displacement_quantity]
                upper_internal_force_array = self.internal_force_array[:self.unknown_displacement_quantity]
                upper_stiffness_matrix = self.stiffness_matrix[:self.unknown_displacement_quantity,
                                         :self.unknown_displacement_quantity]
                residual = self.calculate_residual(external_force_array=upper_external_force_array,
                                                   internal_force_array=upper_internal_force_array)
                # Solve for the unknown displacements u = K^-1*(residual)
                stiffness_matrix_inverse = numpy.linalg.inv(upper_stiffness_matrix)
                self.unknown_displacements = numpy.dot(stiffness_matrix_inverse, residual)
                # Update model configuration for the new displacements
                self.update_current_configuration()
            # Save the maximum deflection in the transverse direction and the load size
            max_deflection = 0
            for node in self.nodes:
                node_deflection = abs(node.current_position[2])
                if node_deflection > max_deflection:
                    max_deflection = node_deflection
            self.maximum_deflections.append(max_deflection)
            self.load_steps.append(abs(current_load[2]))
            # Update the membrane plot
            self.update_plot()

    def rearrange_global(self):
        """Rearrange global quantities such that the rows and columns containing prescribed (known) degrees of
        freedom are moved to the end. We do this so the unknown degrees of freedom will be together on top so that
        they can be solved all at once."""
        # Counter for how many row/columns have been moved
        moved_entries_quantity = 0
        for node in self.nodes:
            for dof_index in range(self.degrees_of_freedom):
                # If there is a known displacement for this degree of freedom, move the row and column to end
                if node.prescribed_displacements[dof_index] is not None:
                    # After each move the remaining row and columns will move up and over one, therefore we need to
                    # adjust the location by 1 * number of moves.
                    index = self.degrees_of_freedom * node.global_id + dof_index - moved_entries_quantity
                    # Move entry in force arrays to end
                    # Save entry to move
                    move_entry_internal = self.internal_force_array[index]
                    # Delete the entry
                    self.internal_force_array = numpy.delete(self.internal_force_array, index)
                    # Insert it at the bottom
                    self.internal_force_array = numpy.append(self.internal_force_array, move_entry_internal)
                    # Move row and column in stiffness matrix to end
                    # Save row to move
                    move_row = self.stiffness_matrix[index]
                    # Delete the row
                    self.stiffness_matrix = numpy.delete(self.stiffness_matrix, index, axis=0)
                    # Insert row at the bottom
                    self.stiffness_matrix = numpy.vstack((self.stiffness_matrix, move_row))
                    # Save column to move
                    move_column = self.stiffness_matrix[index]
                    # Delete the column
                    self.stiffness_matrix = numpy.delete(self.stiffness_matrix, index, axis=1)
                    # Change shape to column (it is a row by default)
                    move_column.shape = (move_column.size, 1)
                    # Insert column at the right
                    self.stiffness_matrix = numpy.hstack((self.stiffness_matrix, move_column))
                    # Increment moved entries counter
                    moved_entries_quantity += 1

    def rearrange_global_external_force_array(self):
        """Rearrange the global external force array such that the rows containing prescribed (known) degrees of
        freedom are moved to the end. We do this so the unknown degrees of freedom will be together on top so that
        they can be solved all at once."""
        # Counter for how many row/columns have been moved
        moved_entries_quantity = 0
        for node in self.nodes:
            for dof_index in range(self.degrees_of_freedom):
                # If there is a known displacement for this degree of freedom, move the row and column to end
                if node.prescribed_displacements[dof_index] is not None:
                    # After each move the remaining row and columns will move up and over one, therefore we need to
                    # adjust the location by 1 * number of moves.
                    index = self.degrees_of_freedom * node.global_id + dof_index - moved_entries_quantity
                    # Move entry in force arrays to end
                    # Save entry to move
                    move_entry = self.external_force_array[index]
                    # Delete the entry
                    self.external_force_array = numpy.delete(self.external_force_array, index)
                    # Insert it at the bottom
                    self.external_force_array = numpy.append(self.external_force_array, move_entry)
                    # Increment moved entries counter
                    moved_entries_quantity += 1

    def run(self):
        """Run the analysis."""
        self.create_mesh()
        self.calculate_node_and_dof_quantities()
        self.create_quadrature_points()
        if self.solve_loading_problem:
            self.loading_solver()
        elif self.solve_displacement_problem:
            self.displacement_solver()
        self.output_results()

    def update_current_configuration(self):
        """Update the current configuration of all elements in the model and assemble the global quantities."""
        # Update the current positions of the nodes with the current values for the unknown displacements
        # Start index counter for progress through unknown displacements
        self.update_node_positions()
        # Update the configuration of the elements (response at quadrature points and integrated element response)
        for element in self.elements:
            element.update_current_configuration()
        # Update the global strain energy, internal force, and stiffness matrix
        self.update_global()

    def update_global(self):
        """Update the global strain energy, internal force, and stiffness matrix from all elements."""
        self.global_strain_energy()
        self.global_internal_force_array()
        self.global_stiffness_matrix()

    def update_node_positions(self):
        """Update the current positions of the nodes."""
        current_index = 0
        for node in self.nodes:
            for dof_index in range(self.degrees_of_freedom):
                # If there is not a prescribed displacement, update the current position for the degree of freedom
                # of the node
                if node.prescribed_displacements[dof_index] is None:
                    node.current_position[dof_index] += self.unknown_displacements[current_index]
                    # Increment the current index
                    current_index += 1

    def update_plot(self):
        """Update the 3D plot for the body."""
        fig = plt.figure(1)
        plt.cla()
        ax = fig.gca(projection='3d')
        # ax.set_zlim(-.015, 0)
        ax.set_xlim(0, .15)
        ax.set_ylim(0, .15)
        x_positions = []
        y_positions = []
        z_positions = []
        for node in self.nodes:
            x_positions.append(node.current_position[0])
            y_positions.append(node.current_position[1])
            z_positions.append(node.current_position[2])
        ax.plot_trisurf(x_positions, y_positions, z_positions, triangles=self.connectivity_table, alpha=.5)
        plt.draw()
        plt.show(block=False)
