"""
elements.py module contains classes for each finite element.

.. moduleauthor:: Tyler Ryan <tyler.ryan@engineering.ucla.edu>
"""
import numpy

import exceptions
import quadrature


class BaseElement:
    """Base element class containing the basic attributes and functions for a finite element. The class
    attributes should be overriden by children classes.

    Finite elements discretize the domain of the body and describe the kinematic and kinetic behavior of a small
    region.
    """
    dimension = 0
    node_quantity = 0
    node_positions = []

    # TODO add arguments to init everywhere
    def __init__(self, material, constitutive_model, quadrature_class, degrees_of_freedom, thickness, plane_stress):
        # Fixed properties
        self.material = material
        self.constitutive_model = constitutive_model
        self.quadrature_class = quadrature_class
        self.degrees_of_freedom = degrees_of_freedom
        self.thickness = thickness
        self.plane_stress = plane_stress

        # Node and quadrature point objects
        self.nodes = []
        self.quadrature_points = []

        # Properties that change with each deformation
        self.strain_energy = None
        self.force_array = None
        self.stiffness_matrix = None

        # Call one-time methods
        self.create_quadrature_points()

    def create_quadrature_points(self):
        """Create quadrature points from the quadrature class."""
        for point_index in range(self.quadrature_class.point_quantity):
            quadrature_point = quadrature.QuadraturePoint(position=self.quadrature_class.point_positions[point_index],
                                                          weight=self.quadrature_class.weights[point_index],
                                                          element=self)
            self.quadrature_points.append(quadrature_point)

    @classmethod
    def shape_functions(cls, node_index, position):
        """Should be overriden by children classes."""
        pass

    @classmethod
    def shape_function_derivatives(cls, node_index, position, coordinate_index):
        """Should be overriden by children classes."""
        pass

    def update_current_configuration(self):
        """Update the positions of the node for the current deformation state, and then compute strain energy,
        the internal nodal force array, and the stiffness matrix using Gauss quadrature."""
        # Update the deformed positions of each node
        for node in self.nodes:
            node.update_current_position()
        # Update the deformation gradient for each quadrature point
        for quadrature_point in self.quadrature_points:
            quadrature_point.update_deformation_gradient(element=self)
            quadrature_point.update_material_response(constitutive_model=self.constitutive_model,
                                                      material=self.material,
                                                      element=self)
        # Update the element properties
        self.update_strain_energy()
        self.update_force_array()
        self.update_stiffness_matrix()

    def update_force_array(self):
        """Computes the 2D internal nodal force array for the element for the current configuration using Gauss
        quadrature. Runs for each deformed configuration in the analysis."""
        # Initialize force array to be computed using Gauss quadrature
        force_array = numpy.zeros((self.degrees_of_freedom, self.node_quantity))
        # Sum over quadrature points
        for quadrature_point in self.quadrature_points:
            # Initialize integrand to be computed for this quadrature point
            integrand = numpy.zeros((self.degrees_of_freedom, self.node_quantity))
            for dof_1 in range(self.degrees_of_freedom):
                for node_index in range(self.node_quantity):
                    # Sum over repeated indices
                    for dof_2 in range(self.degrees_of_freedom):
                        for coordinate_index in range(self.dimension):
                            integrand[dof_1][node_index] += (
                                quadrature_point.first_piola_kirchhoff_stress * self.shape_function_derivatives(
                                    node_index=node_index, position=quadrature_point.position,
                                    coordinate_index=coordinate_index)
                                * quadrature_point.jacobian_inverse[dof_1][coordinate_index])
            # Weight the integrand
            integrand *= quadrature_point.weight
            # Add the integrand to the force_array
            force_array += integrand
        # Scale the force array for isoparametric triangle and multiply by the thickness (assumed to be constant)
        force_array *= .5 * self.thickness
        self.force_array = force_array

    def update_strain_energy(self):
        """Computes the total strain energy of element using Gauss quadrature. Runs for each deformed configuration in
        the analysis."""
        strain_energy = .5 * self.thickness * sum(
            [quadrature_point.strain_energy_density * quadrature_point.weight for quadrature_point in
             self.quadrature_points])
        self.strain_energy = strain_energy

    def update_stiffness_matrix(self):
        """Computes the 4-D stiffness tensor for the element using Gauss quadrature. Runs for each deformed
        configuration in the analysis."""
        # Initialize stiffness matrix to be computed using Gauss quadrature
        dimensions = (self.degrees_of_freedom, self.node_quantity, self.degrees_of_freedom, self.node_quantity)
        stiffness_matrix = numpy.zeros(dimensions)
        # Sum over quadrature points
        for quadrature_point in self.quadrature_points:
            # Initialize integrand to be computed for this quadrature point
            integrand = numpy.zeros(dimensions)
            for dof_1 in range(self.degrees_of_freedom):
                for node_index_1 in range(self.node_quantity):
                    for dof_3 in range(self.degrees_of_freedom):
                        for node_index_2 in range(self.node_quantity):
                            # Sum over repeated indices
                            for dof_2 in range(self.degrees_of_freedom):
                                for dof_4 in range(self.degrees_of_freedom):
                                    for coordinate_index_1 in range(self.dimension):
                                        for coordinate_index_2 in range(self.dimension):
                                            integrand[dof_1][node_index_1][dof_3][node_index_2] += (
                                                quadrature_point.tangent_moduli
                                                * self.shape_function_derivatives(
                                                    node_index=node_index_1,
                                                    position=quadrature_point.position,
                                                    coordinate_index=coordinate_index_1)
                                                * self.shape_function_derivatives(
                                                    node_index=node_index_2,
                                                    position=quadrature_point.position,
                                                    coordinate_index=coordinate_index_2)
                                                * quadrature_point.jacobian_inverse[dof_2][coordinate_index_1]
                                                * quadrature_point.jacobian_inverse[dof_4][coordinate_index_2])
            # Weight the integrand
            integrand *= quadrature_point.weight
            # Add the integrand to the stiffness matrix
            stiffness_matrix += integrand
        # Scale the stiffness matrix for isoparametric triangle and multiply by the thickness (assumed to be constant)
        stiffness_matrix *= .5 * self.thickness
        self.stiffness_matrix = stiffness_matrix


class TriangularLinearElement(BaseElement):
    """A 2-D isoparametric triangular element with 3 nodes."""
    dimension = 2
    node_quantity = 3
    node_positions = [(0, 0), (1, 0), (0, 1)]

    def __init__(self):
        super(BaseElement, self).__init__()

    @classmethod
    def shape_functions(cls, node_index, position):
        """Compute the value of the shape function for the specified node_index at the given coordinates.
        Note that the nodes are indexed starting at 0 for the convenience of iteration when calling this function.

        :param int node_index: index of the node_index at which to compute the shape function
        :param tuple position: coordinates of point at which to evaluate
        """
        if node_index == 0:
            return 1 - position[0] - position[1]
        elif node_index == 1:
            return position[0]
        elif node_index == 2:
            return position[1]
        else:
            exceptions.InvalidNodeError(node_index=node_index, node_quantity=cls.node_quantity)

    @classmethod
    def shape_function_derivatives(cls, node_index, position, coordinate_index):
        """Compute the value of the derivative of the shape function with respect to the specified coordinate,
        for the specified node_index at the given coordinates. Note that the nodes and coordinates are indexed starting
        at 0 for the convenience of iteration when calling this function.

        :param int node_index: index of the node at which to compute the shape function
        :param tuple position: coordinates of point at which to evaluate
        :param int coordinate_index: index of the coordinate to compute the derivative with respect to
        """
        if node_index == 0:
            if coordinate_index == 0:
                return -1
            elif coordinate_index == 1:
                return -1
            else:
                raise exceptions.InvalidCoordinateError(coordinate_index=coordinate_index,
                                                        coordinate_quantity=cls.dimension)
        elif node_index == 1:
            if coordinate_index == 0:
                return 1
            elif coordinate_index == 1:
                return 0
            else:
                raise exceptions.InvalidCoordinateError(coordinate_index=coordinate_index,
                                                        coordinate_quantity=cls.dimension)
        elif node_index == 2:
            if coordinate_index == 0:
                return 0
            elif coordinate_index == 1:
                return 1
            else:
                raise exceptions.InvalidCoordinateError(coordinate_index=coordinate_index,
                                                        coordinate_quantity=cls.dimension)
        else:
            raise exceptions.InvalidNodeError(node_index=node_index, node_quantity=cls.node_quantity)


class TriangularQuadraticElement(BaseElement):
    """A 2-D isoparametric triangular element with 6 nodes."""
    dimension = 2
    node_quantity = 6
    node_positions = [(0, 0), (1, 0), (0, 1), (.5, 0), (.5, .5), (0, .5)]

    def __init__(self):
        super(BaseElement, self).__init__()

    @classmethod
    def shape_functions(cls, node_index, position):
        """Compute the value of the shape function for the specified node_index at the given coordinates.
        Note that the nodes are indexed starting at 0 for the convenience of iteration when calling this function.

        :param int node_index: index of the node_index at which to compute the shape function
        :param tuple position: coordinates of point at which to evaluate
        """
        if node_index == 0:
            return 2 * (1 - position[0] - position[1]) * (.5 - position[0] - position[1])
        elif node_index == 1:
            return 2 * position[0] * (position[0] - .5)
        elif node_index == 2:
            return 2 * position[1] * (position[1] - .5)
        elif node_index == 3:
            return 4 * position[0] * (1 - position[0] - position[1])
        elif node_index == 4:
            return 4 * position[0] * position[1]
        elif node_index == 5:
            return 4 * position[1] * (1 - position[0] - position[1])
        else:
            exceptions.InvalidNodeError(node_index=node_index, node_quantity=cls.node_quantity)

    @classmethod
    def shape_function_derivatives(cls, node_index, position, coordinate_index):
        """Compute the value of the derivative of the shape function with respect to the specified coordinate,
        for the specified node_index at the given coordinates. Note that the nodes and coordinates are indexed starting
        at 0 for the convenience of iteration when calling this function.

        :param int node_index: index of the node at which to compute the shape function
        :param tuple position: coordinates of point at which to evaluate
        :param int coordinate_index: index of the coordinate to compute the derivative with respect to
        """
        if node_index == 0:
            if coordinate_index == 0:
                return 4 * position[0] + 4 * position[1] - 3
            elif coordinate_index == 1:
                return 4 * position[0] + 4 * position[1] - 3
            else:
                raise exceptions.InvalidCoordinateError(coordinate_index=coordinate_index,
                                                        coordinate_quantity=cls.dimension)
        elif node_index == 1:
            if coordinate_index == 0:
                return 4 * position[0] - 1
            elif coordinate_index == 1:
                return 0
            else:
                raise exceptions.InvalidCoordinateError(coordinate_index=coordinate_index,
                                                        coordinate_quantity=cls.dimension)
        elif node_index == 2:
            if coordinate_index == 0:
                return 0
            elif coordinate_index == 1:
                return 4 * position[1] - 1
            else:
                raise exceptions.InvalidCoordinateError(coordinate_index=coordinate_index,
                                                        coordinate_quantity=cls.dimension)
        elif node_index == 3:
            if coordinate_index == 0:
                return -8 * position[0] - 4 * position[1] + 4
            elif coordinate_index == 1:
                return -4 * position[0]
            else:
                raise exceptions.InvalidCoordinateError(coordinate_index=coordinate_index,
                                                        coordinate_quantity=cls.dimension)
        elif node_index == 4:
            if coordinate_index == 0:
                return 4 * position[1]
            elif coordinate_index == 1:
                return 4 * position[0]
            else:
                raise exceptions.InvalidCoordinateError(coordinate_index=coordinate_index,
                                                        coordinate_quantity=cls.dimension)
        elif node_index == 5:
            if coordinate_index == 0:
                return -4 * position[1]
            elif coordinate_index == 1:
                return -4 * position[0] - 8 * position[1] + 4
            else:
                raise exceptions.InvalidCoordinateError(coordinate_index=coordinate_index,
                                                        coordinate_quantity=cls.dimension)
        else:
            raise exceptions.InvalidNodeError(node_index=node_index, node_quantity=cls.node_quantity)
