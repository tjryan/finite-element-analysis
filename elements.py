"""
elements.py module contains classes for each finite element.

.. moduleauthor:: Tyler Ryan <tyler.ryan@engineering.ucla.edu>
"""

import exceptions


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
    def __init__(self):
        self.nodes = []
        self.quadrature_points = []
        self.jacobian_matrix = None
        self.jacobian_inverse_transpose = None
        self.reference_configuration = None  # matrix with node reference positions as row vectors
        self.current_configuration = None  # matrix with node current positions as row vector
        # Call one-time methods
        self.assemble_reference_configuration()
        self.calculate_jacobian_matrix()

    def assemble_current_configuration(self):
        """Assemble the current configuration matrix from the node current positions as row vectors.
        Runs one time and remains constant for the element throughout the analysis."""
        # self.current_configuration = something
        pass

    def assemble_reference_configuration(self):
        """Assemble the reference configuration matrix from the node reference positions as row vectors.
        Runs one time and remains constant for the element throughout the analysis."""
        # self.reference_configuration = something
        pass

    def calculate_jacobian_matrix(self):
        """Computes the Jacobian matrix and its inverse-transpose for the element.
        Runs one time and remains constant for the element throughout the analysis."""
        # self.jacobian_matrix = something
        # self.jacobian_inverse_transpose = somethingelse
        pass

    def force_array(self):
        """Computes the 2D internal nodal force array for the element for the current configuration.
        Runs for each deformed configuration in the analysis."""
        # return force_array
        pass

    def stiffness_tensor(self):
        """Computes the 4-D stiffness tensor for the element. Runs for each deformed configuration in the analysis."""
        # return stiffness tensor
        pass

    def strain_energy(self):
        """Computes the total strain energy of element. Runs for each deformed configuration in the analysis."""
        pass


class TriangularLinearElement(BaseElement):
    """A 2-D isoparametric triangular element with 3 nodes."""
    dimension = 2
    node_quantity = 3
    node_positions = [(0, 0), (1, 0), (0, 1)]

    def __init__(self):
        super(BaseElement, self).__init__()

    @classmethod
    def shape_functions(cls, node_index, r, s):
        """Compute the value of the shape function for the specified node_index at the given coordinates.
        Note that the nodes are indexed starting at 0 for the convenience of iteration when calling this function.

        :param int node_index: index of the node_index at which to compute the shape function
        :param float r: value of first coordinate
        :param float s: value of second coordinate
        """
        if node_index == 0:
            return 1 - r - s
        elif node_index == 1:
            return r
        elif node_index == 2:
            return s
        else:
            exceptions.InvalidNodeError(node_index=node_index, node_quantity=cls.node_quantity)

    @classmethod
    def shape_function_derivatives(cls, node_index, r, s, coordinate_index):
        """Compute the value of the derivative of the shape function with respect to the specified coordinate,
        for the specified node_index at the given coordinates. Note that the nodes and coordinates are indexed starting
        at 0 for the convenience of iteration when calling this function.

        :param int node_index: index of the node at which to compute the shape function
        :param float r: value of first coordinate
        :param float s: value of second coordinate
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
    def shape_functions(cls, node_index, r, s):
        """Compute the value of the shape function for the specified node_index at the given coordinates.
        Note that the nodes are indexed starting at 0 for the convenience of iteration when calling this function.

        :param int node_index: index of the node_index at which to compute the shape function
        :param float r: value of first coordinate
        :param float s: value of second coordinate
        """
        if node_index == 0:
            return 2 * (1 - r - s) * (.5 - r - s)
        elif node_index == 1:
            return 2 * r * (r - .5)
        elif node_index == 2:
            return 2 * s * (s - .5)
        elif node_index == 3:
            return 4 * r * (1 - r - s)
        elif node_index == 4:
            return 4 * r * s
        elif node_index == 5:
            return 4 * s * (1 - r - s)
        else:
            exceptions.InvalidNodeError(node_index=node_index, node_quantity=cls.node_quantity)

    @classmethod
    def shape_function_derivatives(cls, node_index, r, s, coordinate_index):
        """Compute the value of the derivative of the shape function with respect to the specified coordinate,
        for the specified node_index at the given coordinates. Note that the nodes and coordinates are indexed starting
        at 0 for the convenience of iteration when calling this function.

        :param int node_index: index of the node at which to compute the shape function
        :param float r: value of first coordinate
        :param float s: value of second coordinate
        :param int coordinate_index: index of the coordinate to compute the derivative with respect to
        """
        if node_index == 0:
            if coordinate_index == 0:
                return 4 * r + 4 * s - 3
            elif coordinate_index == 1:
                return 4 * r + 4 * s - 3
            else:
                raise exceptions.InvalidCoordinateError(coordinate_index=coordinate_index,
                                                        coordinate_quantity=cls.dimension)
        elif node_index == 1:
            if coordinate_index == 0:
                return 4 * r - 1
            elif coordinate_index == 1:
                return 0
            else:
                raise exceptions.InvalidCoordinateError(coordinate_index=coordinate_index,
                                                        coordinate_quantity=cls.dimension)
        elif node_index == 2:
            if coordinate_index == 0:
                return 0
            elif coordinate_index == 1:
                return 4 * s - 1
            else:
                raise exceptions.InvalidCoordinateError(coordinate_index=coordinate_index,
                                                        coordinate_quantity=cls.dimension)
        elif node_index == 3:
            if coordinate_index == 0:
                return -8 * r - 4 * s + 4
            elif coordinate_index == 1:
                return -4 * r
            else:
                raise exceptions.InvalidCoordinateError(coordinate_index=coordinate_index,
                                                        coordinate_quantity=cls.dimension)
        elif node_index == 4:
            if coordinate_index == 0:
                return 4 * s
            elif coordinate_index == 1:
                return 4 * r
            else:
                raise exceptions.InvalidCoordinateError(coordinate_index=coordinate_index,
                                                        coordinate_quantity=cls.dimension)
        elif node_index == 5:
            if coordinate_index == 0:
                return -4 * s
            elif coordinate_index == 1:
                return -4 * r - 8 * s + 4
            else:
                raise exceptions.InvalidCoordinateError(coordinate_index=coordinate_index,
                                                        coordinate_quantity=cls.dimension)
        else:
            raise exceptions.InvalidNodeError(node_index=node_index, node_quantity=cls.node_quantity)
