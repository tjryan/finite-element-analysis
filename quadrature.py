"""
quadrature.py contains Gauss quadrature tables for use in numerical integration.

.. moduleauthor:: Tyler Ryan <tyler.ryan@engineering.ucla.edu>
"""


class BaseQuadrature:
    """Base class for numerical integration of isoparametric triangular elements.
    Attributes should be overriden by child classes."""

    point_quantity = 0
    point_positions = []
    point_weights = []

    @classmethod
    def integrate(cls, function):
        """Use Gauss quadrature to numerically integrate a function.

        :param function: a lambda function of two variables to integrate numerically
        """
        result = .5 * sum(
            [function(cls.point_positions[point_index][0], cls.point_positions[point_index][1]) * cls.point_weights[
                point_index] for
             point_index in range(cls.point_quantity)])
        return result


class GaussQuadratureOnePoint(BaseQuadrature):
    """Properties of one-point Gauss quadrature."""

    point_quantity = 1
    point_positions = [(1 / 3, 1 / 3)]
    point_weights = [1]


class GaussQuadratureThreePoint(BaseQuadrature):
    """Properties of three-point Gauss quadrature."""

    point_quantity = 3
    point_positions = [(1 / 6, 1 / 6), (2 / 3, 1 / 6), (1 / 6, 2 / 3)]
    point_weights = [1 / 3, 1 / 3, 1 / 3]

