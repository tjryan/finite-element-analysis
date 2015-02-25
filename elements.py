"""
elements.py module contains classes for each finite element.

.. moduleauthor:: Tyler Ryan <tyler.ryan@engineering.ucla.edu>
"""

import exceptions


class TriangularElementThreeNode:
    """A 2-D isoparametric triangular element with 3 nodes."""
    dimension = 2
    node_quantity = 3
    node_locations = [(0, 0), (1, 0), (0, 1)]

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


class TriangularElementSixNode:
    """A 2-D isoparametric triangular element with 6 nodes."""
    dimension = 2
    node_quantity = 6
    node_locations = [(0, 0), (1, 0), (0, 1), (.5, 0), (.5, .5), (0, .5)]

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
