"""
material.py contains the material model for the body that describes its properties.

.. moduleauthor:: Tyler Ryan <tyler.ryan@engineering.ucla.edu>
"""

import numpy

import operations
import tests


class DeformationGradient:
    """A 3x3 matrix that describes the deformation of the body.

    :ivar numpy.ndarray F: 3x3 tensor describing deformation
    :ivar float J: Jacobian, representing the determinant of the deformation gradient tensor
    """

    def __init__(self):
        self.F = None
        self.J = None

    def calculate_jacobian(self):
        """Compute the value for the Jacobian when the deformation gradient is initialized,
        and confirm that it has a positive value. If the Jacobian is negative, the element is likely too
        distorted, so raise an error.
        """
        jacobian = numpy.linalg.det(self.F)
        tests.deformation_gradient_physical(jacobian=jacobian)
        return jacobian

    def enforce_plane_stress(self, material, constitutive_model):
        """Enforce the plane stress assumption solving for the thickness stretch (the 3-3 component of the
        deformation gradient), while assuming that the  first Piola-Kirchhoff stress is zero in the direction
        normal to the plane.

        Note: this function assumes that the deformation gradient already has the following form:
                        F = [F11 F12  0 ]
                            [F21 F22  0 ]
                            [ 0   0  F33]
        This will be checked before performing any calculations, and an error will be raised if violated.

        :param material: material object representing the material undergoing deformation
        :param constitutive_model: constitutive model class that described material response
        """
        # Check that the deformation gradient has the proper structure for plane stress
        tests.deformation_gradient_plane_stress(deformation_gradient=self)
        thickness_stretch_ratio = operations.newton_method_thickness_stretch_ratio(material=material,
                                                                                   constitutive_model=constitutive_model,
                                                                                   deformation_gradient=self.F)
        # Assign the 3-3 component of the deformation gradient to be the computed thickness stretch ratio
        self.F[2][2] = thickness_stretch_ratio
        # Update the Jacobian for the new deformation gradient
        self.J = self.calculate_jacobian()

    def update_F(self, new_F, material, constitutive_model, enforce_plane_stress):
        """Update the deformation gradient matrix with the newly provided matrix and enforce plane stress
        if necessary.

        :param numpy.ndarray new_F: new 3x3 deformation gradient matrix
        :param material: material object representing the material undergoing deformation
        :param constitutive_model: constitutive model class that described material response
        :param bool enforce_plane_stress: whether to enforce the plane stress condition on the new deformation gradient
        """
        self.F = new_F
        if enforce_plane_stress:
            self.enforce_plane_stress(material=material, constitutive_model=constitutive_model)
        self.J = self.calculate_jacobian()


