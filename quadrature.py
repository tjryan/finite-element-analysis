"""
quadrature.py contains Gauss quadrature tables for use in numerical integration.

.. moduleauthor:: Tyler Ryan <tyler.ryan@engineering.ucla.edu>
"""
import numpy

import deformation_gradient


class BaseQuadrature:
    """Base class for numerical integration of isoparametric triangular elements.
    Attributes should be overriden by child classes.

    :cvar int point_quantity: number of quadrature points
    :cvar list point_positions: list of tuples of quadrature point positions
    :cvar list point_weights: list of quadrature point weights
    """

    point_quantity = 0
    point_positions = []
    point_weights = []


class GaussQuadratureOnePoint(BaseQuadrature):
    """Properties of one-point Gauss quadrature.

    :cvar int point_quantity: number of quadrature points
    :cvar list point_positions: list of tuples of quadrature point positions
    :cvar list point_weights: list of quadrature point weights
    """

    point_quantity = 1
    point_positions = [(1 / 3, 1 / 3)]
    point_weights = [1]


class GaussQuadratureThreePoint(BaseQuadrature):
    """Properties of three-point Gauss quadrature.

    :cvar int point_quantity: number of quadrature points
    :cvar list point_positions: list of tuples of quadrature point positions
    :cvar list point_weights: list of quadrature point weights
    """

    point_quantity = 3
    point_positions = [(1 / 6, 1 / 6), (2 / 3, 1 / 6), (1 / 6, 2 / 3)]
    point_weights = [1 / 3, 1 / 3, 1 / 3]


class QuadraturePoint:
    """Individual quadrature point containing material responses at a particular point in an isoparametric element.

    :param tuple position: coordinates of quadrature point position
    :param float weight: weight of quadrature point
    :param element: element object containing the quadrature point
    :ivar numpy.ndarray jacobian_matrix: jacobian matrix describing the mapping of the reference configuration to the
    isoparametric configuration at the quadrature point. Remains constant for the whole analysis.
    :ivar numpy.ndarray jacobian_matrix_inverse: inverse of the jacobian matrix. Remains constant for the whole
    analysis.
    :ivar deformation_gradient: DeformationGradient object that describes the deformation for the current configuration
    as the quadrature point. Updates with every step.
    :ivar float strain_energy_density: strain energy density at the quadrature point
    :ivar numpy.ndarray first_piola_kirchhoff_stress: first Piola-Kirchhoff stress at the quadrature point
    :ivar numpy.ndarray tangent_moduli: tangent moduli at the quadrature point
    """

    def __init__(self, position, weight, element):
        self.position = position
        self.weight = weight

        # Updated every deformation
        self.deformation_gradient = deformation_gradient.DeformationGradient()
        self.strain_energy_density = None
        self.first_piola_kirchhoff_stress = None
        self.tangent_moduli = None

    def update_deformation_gradient(self, element):
        """Update the deformation gradient object for the current deformation.

        :param element: element object that is deformed
        """
        # Deformation gradient initialized as a 3x3 matrix, always
        new_deformation_gradient = numpy.zeros((3, 3))
        for dof_1 in range(element.degrees_of_freedom):
            for dof_2 in range(element.degrees_of_freedom):
                # Sum over nodes to interpolate value at this quadrature point
                for node_index in range(element.node_quantity):
                    for coordinate_index in range(element.dimension):
                        new_deformation_gradient[dof_1][dof_2] += (
                            element.nodes[node_index].current_position[dof_1] * element.shape_function_derivatives(
                                node_index=node_index, position=self.position, coordinate_index=coordinate_index)
                            * element.jacobian_matrix_inverse[coordinate_index][dof_2])
        if element.plane_stress:
            new_deformation_gradient[2][2] = 1
        self.deformation_gradient.update_F(new_F=new_deformation_gradient,
                                           material=element.material,
                                           constitutive_model=element.constitutive_model,
                                           enforce_plane_stress=element.plane_stress)

    def update_material_response(self, element):
        """Update the strain energy density, first Piola-Kirchhoff stress and tangent moduli for the current
        configuration.

        :param element: element object that is deformed
        """
        (self.strain_energy_density,
         self.first_piola_kirchhoff_stress,
         self.tangent_moduli) = element.constitutive_model.calculate_all(
            material=element.material,
            deformation_gradient=self.deformation_gradient.F,
            dimension=element.dimension)