"""
elements.py module contains classes for each finite element.

.. moduleauthor:: Tyler Ryan <tyler.ryan@engineering.ucla.edu>
"""
import numpy

import exceptions
import quadrature
import tests


class BaseElement:
    """Base element class containing the basic attributes and functions for a finite element. The class
    attributes should be overriden by children classes.

    Finite elements discretize the domain of the body and describe the kinematic and material response of a small
    region.

    :cvar int dimension: the dimension of the element (1D, 2D, 3D)
    :cvar int node_quantity: the number of nodes in the element
    :cvar list node_positions: a list of tuples containing the positions of the nodes
    :param constitutive_model: constitutive model class that describes the material behavior
    :param material: material object that described the material the element is composed of
    :param quadrature_class: class that describes the order of quadrature being used
    :param int degrees_of_freedom: number of degrees of freedom at each node of the element
    :param float thickness: thickness of the element (assumed to be constant over the element)
    :ivar list nodes: list of node objects contained in the element
    :ivar list quadrature_points: list of quadrature point objects contained in the element
    :ivar float strain_energy: strain energy of the element
    :ivar numpy.ndarray force_array: 2D matrix containing the effective forces on each node of the element
    :ivar numpy.ndarray stiffness_matrix: 4D matrix describing the stiffness of the element against deformation
    """
    dimension = 0
    node_quantity = 0
    node_positions = []

    def __init__(self, constitutive_model, material, quadrature_class, degrees_of_freedom, thickness):
        # Fixed properties
        self.constitutive_model = constitutive_model
        self.material = material
        self.quadrature_class = quadrature_class
        self.degrees_of_freedom = degrees_of_freedom
        self.thickness = thickness

        # Node and quadrature point objects
        self.nodes = []
        self.quadrature_points = []

        # Calculated one time
        self.jacobian_matrix = None
        self.jacobian_matrix_inverse = None

        # Properties that change with each deformation
        self.strain_energy = None
        self.force_array = None
        self.stiffness_matrix = None

    def calculate_force_array(self, test=False):
        """Computes the 2D internal nodal force array for the element for the current configuration using Gauss
       quadrature. Runs for each deformed configuration in the analysis.

       :param bool test: whether to perform numerical differentiation check on the result
       """
        # Initialize force array to be computed using Gauss quadrature
        force_array = numpy.zeros((self.degrees_of_freedom, self.node_quantity))  # Sum over quadrature points
        for quadrature_point in self.quadrature_points:
            # Initialize integrand to be computed for this quadrature point
            integrand = numpy.zeros((self.degrees_of_freedom, self.node_quantity))
            for dof_1 in range(self.degrees_of_freedom):
                for node_index in range(self.node_quantity):
                    # Sum over repeated indices
                    for dof_2 in range(self.degrees_of_freedom):
                        for coordinate_index in range(self.dimension):
                            integrand[dof_1][node_index] += (
                                quadrature_point.first_piola_kirchhoff_stress[dof_1][dof_2]
                                * self.shape_function_derivatives(
                                    node_index=node_index, position=quadrature_point.position,
                                    coordinate_index=coordinate_index)
                                * self.jacobian_matrix_inverse[coordinate_index][dof_2])
            # Weight the integrand
            integrand *= quadrature_point.weight
            # Add the integrand to the force_array
            force_array += integrand
        # Scale the force array for isoparametric triangle and multiply by the thickness (assumed to be constant)
        force_array *= .5 * self.thickness
        if test:
            tests.numerical_differentiation_force_array(element=self, force_array=force_array)
        return force_array

    def calculate_jacobian_matrix(self):
        """Calculate the Jacobian matrix for the element. This is a one time calculation performed during the
        creation of the quadrature points.
        """
        quadrature_point = self.quadrature_points[0]
        jacobian_matrix = numpy.zeros((self.degrees_of_freedom, self.dimension))
        for dof in range(self.degrees_of_freedom):
            for coordinate_index in range(self.dimension):
                for node_index in range(self.node_quantity):
                    jacobian_matrix[dof][coordinate_index] += (
                        self.nodes[node_index].reference_position[dof] * self.shape_function_derivatives(
                            node_index=node_index, position=quadrature_point.position,
                            coordinate_index=coordinate_index))
        self.jacobian_matrix = jacobian_matrix
        # self.jacobian_matrix_inverse = numpy.linalg.inv(jacobian_matrix)

    def calculate_strain_energy(self):
        """Computes the total strain energy of element using Gauss quadrature. Runs for each deformed configuration in
        the analysis."""
        strain_energy = .5 * self.thickness * sum(
            [quadrature_point.strain_energy_density * quadrature_point.weight for quadrature_point in
             self.quadrature_points])
        return strain_energy

    def calculate_stiffness_matrix(self, test=False, rank=True):
        """Computes the 4-D stiffness tensor for the element using Gauss quadrature. Runs for each deformed
        configuration in the analysis.

        :param bool test: whether to perform numerical differentiation check on the result
        :param bool rank: whether to check the rank of the stiffness matrix
        """
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
                                                quadrature_point.tangent_moduli[dof_1][dof_2][dof_3][dof_4]
                                                * self.shape_function_derivatives(
                                                    node_index=node_index_1,
                                                    position=quadrature_point.position,
                                                    coordinate_index=coordinate_index_1)
                                                * self.shape_function_derivatives(
                                                    node_index=node_index_2,
                                                    position=quadrature_point.position,
                                                    coordinate_index=coordinate_index_2)
                                                * self.jacobian_matrix_inverse[coordinate_index_1][dof_2]
                                                * self.jacobian_matrix_inverse[coordinate_index_2][dof_4])
            # Weight the integrand
            integrand *= quadrature_point.weight
            # Add the integrand to the stiffness matrix
            stiffness_matrix += integrand
        # Scale the stiffness matrix for isoparametric triangle and multiply by the thickness (assumed to be constant)
        stiffness_matrix *= .5 * self.thickness
        if test:
            tests.numerical_differentiation_stiffness_matrix(element=self, stiffness_matrix=stiffness_matrix)
        if rank:
            tests.rank_stiffness_matrix(element=self, stiffness_matrix=stiffness_matrix)
        return stiffness_matrix

    def create_quadrature_points(self):
        """Create quadrature points from the quadrature class. This is a one-time method called after all nodes
        have been assigned to the element."""
        for point_index in range(self.quadrature_class.point_quantity):
            quadrature_point = quadrature.QuadraturePoint(position=self.quadrature_class.point_positions[point_index],
                                                          weight=self.quadrature_class.point_weights[point_index],
                                                          element=self)
            self.quadrature_points.append(quadrature_point)
        self.calculate_jacobian_matrix()

    def update_current_configuration(self):
        """Update the positions of the node for the current deformation state, and then compute strain energy,
        the internal nodal force array, and the stiffness matrix using Gauss quadrature."""
        # Update the deformed positions of each node
        for node in self.nodes:
            node.update_current_position()
        # Update the deformation gradient for each quadrature point
        for quadrature_point in self.quadrature_points:
            quadrature_point.update_deformation_gradient(element=self)
            quadrature_point.update_material_response(element=self)
        # Update the element properties
        self.update_strain_energy()
        self.update_force_array()
        self.update_stiffness_matrix()

    def update_force_array(self):
        self.force_array = self.calculate_force_array()

    def update_strain_energy(self):
        """Update the strain energy for the current deformation."""
        self.strain_energy = self.calculate_strain_energy()

    def update_stiffness_matrix(self):
        """Update the stiffness matrix for the current deformation."""
        self.stiffness_matrix = self.calculate_stiffness_matrix()

    @classmethod
    def shape_functions(cls, node_index, position):
        """Should be overriden by children classes. Compute the value of the shape function for the specified
        node_index at the given coordinates. Note that the nodes are indexed starting at 0 for the convenience of
        iteration when calling this function.

        :param int node_index: index of the node_index at which to compute the shape function
        :param tuple position: coordinates of point at which to evaluate
        """
        pass

    @classmethod
    def shape_function_derivatives(cls, node_index, position, coordinate_index):
        """Should be overriden by children classes. Compute the value of the derivative of the shape function with
        respect to the specified coordinate, for the specified node_index at the given coordinates. Note that the
        nodes and coordinates are indexed starting at 0 for the convenience of iteration when calling this function.

        :param int node_index: index of the node at which to compute the shape function
        :param tuple position: coordinates of point at which to evaluate
        :param int coordinate_index: index of the coordinate to compute the derivative with respect to
        """
        pass


class TriangularLinearElement(BaseElement):
    """A 2-D isoparametric linear triangular element with 3 nodes.

    :cvar int dimension: the dimension of the element (1D, 2D, 3D)
    :cvar int node_quantity: the number of nodes in the element
    :cvar list node_positions: a list of tuples containing the positions of the nodes
    :param constitutive_model: constitutive model class that describes the material behavior
    :param material: material object that described the material the element is composed of
    :param quadrature_class: class that describes the order of quadrature being used
    :param int degrees_of_freedom: number of degrees of freedom at each node of the element
    :param float thickness: thickness of the element (assumed to be constant over the element)
    :ivar list nodes: list of node objects contained in the element
    :ivar list quadrature_points: list of quadrature point objects contained in the element
    :ivar float strain_energy: strain energy of the element
    :ivar numpy.ndarray force_array: 2D matrix containing the effective forces on each node of the element
    :ivar numpy.ndarray stiffness_matrix: 4D matrix describing the stiffness of the element against deformation
    """

    dimension = 2
    node_quantity = 3
    node_positions = [(0, 0), (1, 0), (0, 1)]

    def __init__(self, constitutive_model, material, quadrature_class, degrees_of_freedom, thickness):
        super(TriangularLinearElement, self).__init__(constitutive_model, material, quadrature_class,
                                                      degrees_of_freedom, thickness)

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
    """A 2-D isoparametric triangular element with 6 nodes.

    :cvar int dimension: the dimension of the element (1D, 2D, 3D)
    :cvar int node_quantity: the number of nodes in the element
    :cvar list node_positions: a list of tuples containing the positions of the nodes
    :param constitutive_model: constitutive model class that describes the material behavior
    :param material: material object that described the material the element is composed of
    :param quadrature_class: class that describes the order of quadrature being used
    :param int degrees_of_freedom: number of degrees of freedom at each node of the element
    :param float thickness: thickness of the element (assumed to be constant over the element)
    :ivar list nodes: list of node objects contained in the element
    :ivar list quadrature_points: list of quadrature point objects contained in the element
    :ivar float strain_energy: strain energy of the element
    :ivar numpy.ndarray force_array: 2D matrix containing the effective forces on each node of the element
    :ivar numpy.ndarray stiffness_matrix: 4D matrix describing the stiffness of the element against deformation
    """

    dimension = 2
    node_quantity = 6
    node_positions = [(0, 0), (1, 0), (0, 1), (.5, 0), (.5, .5), (0, .5)]

    def __init__(self, constitutive_model, material, quadrature_class, degrees_of_freedom, thickness):
        super(TriangularQuadraticElement, self).__init__(constitutive_model, material, quadrature_class,
                                                         degrees_of_freedom, thickness)

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
