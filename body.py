"""
material.py contains the material model for the body that describes its properties.

.. moduleauthor:: Tyler Ryan <tyler.ryan@engineering.ucla.edu>
"""

import numpy

import operations
import tests


class DeformationGradient:
    """Describes the deformation of the body.

    :param numpy.ndarray F: 3x3 tensor describing deformation
    :ivar float J: Jacobian, representing the determinant of the deformation gradient tensor
    """
    # TODO consistent implementation of deformation gradient as a class everywhere

    def __init__(self, deformation_gradient, material, constitutive_model, plane_stress=False, plane_strain=False):
        self.F = deformation_gradient
        self.material = material
        self.constitutive_model = constitutive_model
        # TODO should I store these?
        self.plane_stress = plane_stress
        self.plane_strain = plane_strain
        # Compute the Jacobian and enforce assumptions
        self.J = self.calculate_jacobian()
        if self.plane_stress:
            self.enforce_plane_stress()
        if self.plane_strain:
            self.enforce_plane_strain()

    def calculate_jacobian(self):
        """Compute the value for the Jacobian when the deformation gradient is initialized,
        and confirm that it has a positive value. If the Jacobian is negative, the element is likely too
        distorted, so raise an error.
        """
        jacobian = numpy.linalg.det(self.F)
        tests.check_deformation_gradient_physical(jacobian=jacobian)
        return jacobian

    def enforce_plane_stress(self):
        """Enforce the plane stress assumption solving for the thickness stretch (the 3-3 component of the
        deformation gradient), while assuming that the  first Piola-Kirchhoff stress is zero in the direction
        normal to the plane.

        Note: this function assumes that the deformation gradient already has the following form:
                        F = [F11 F12  0 ]
                            [F21 F22  0 ]
                            [ 0   0  F33]
        This will be checked before performing any calculations, and an error will be raised if violated.
        """
        # Check that the deformation gradient has the proper structure for plane stress
        tests.check_deformation_gradient_plane_stress(deformation_gradient=self.F)
        thickness_stretch_ratio = operations.newton_method_thickness_stretch_ratio(material=self.material,
                                                                                   constitutive_model=self.constitutive_model,
                                                                                   deformation_gradient=self.F)
        # Assign the 3-3 component of the deformation gradient to be the computed thickness stretch ratio
        self.F[2][2] = thickness_stretch_ratio
        # Update the Jacobian for the new deformation gradient
        self.J = self.calculate_jacobian()

    def enforce_plane_strain(self):
        # TODO implement this method
        pass


