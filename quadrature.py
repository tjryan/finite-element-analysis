"""
quadrature.py contains Gauss quadrature tables for use in numerical integration.

.. moduleauthor:: Tyler Ryan <tyler.ryan@engineering.ucla.edu>
"""
import numpy

import operations
import tests


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
        self.deformation_gradient = None
        self.jacobian = None
        self.strain_energy_density = None
        self.first_piola_kirchhoff_stress = None
        self.tangent_moduli = None

    def calculate_jacobian(self):
        """Compute the value for the Jacobian when the deformation gradient is updated,
        and confirm that it has a positive value.
        """
        jacobian = numpy.linalg.det(self.deformation_gradient)
        tests.deformation_gradient_physical(jacobian=jacobian)
        return jacobian

    def enforce_plane_stress(self, element):
        """Enforce the plane stress assumption solving for the thickness stretch (the 3-3 component of the
        deformation gradient), while assuming that the  first Piola-Kirchhoff stress is zero in the direction
        normal to the plane.

        Note: this function assumes that the deformation gradient already has the following form:
                        F = [F11 F12  0 ]
                            [F21 F22  0 ]
                            [ 0   0  F33]
        This will be checked before performing any calculations, and an error will be raised if violated.

        :param element: element object that contains the quadrature point
        """
        # Check that the deformation gradient has the proper structure for plane stress
        tests.deformation_gradient_plane_stress(deformation_gradient=self.deformation_gradient)
        thickness_stretch_ratio = operations.newton_method_thickness_stretch_ratio(
            constitutive_model=element.constitutive_model,
            material=element.material,
            deformation_gradient=self.deformation_gradient)
        # Assign the 3-3 component of the deformation gradient to be the computed thickness stretch ratio
        self.deformation_gradient[2][2] = thickness_stretch_ratio

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
        self.deformation_gradient = new_deformation_gradient
        if element.plane_stress:
            self.enforce_plane_stress(element=element)
        # Update the Jacobian for the new deformation gradient
        self.jacobian = self.calculate_jacobian()


    def update_material_response(self, element):
        """Update the strain energy density, first Piola-Kirchhoff stress and tangent moduli for the current
        configuration.

        :param element: element object that is deformed
        """
        (self.strain_energy_density,
         self.first_piola_kirchhoff_stress,
         self.tangent_moduli) = element.constitutive_model.calculate_all(
            material=element.material,
            deformation_gradient=self.deformation_gradient,
            dimension=element.dimension)