"""
kinematics.py contains functions for computing kinematic quantities.

.. moduleauthor:: Tyler Ryan <tyler.ryan@engineering.ucla.edu>
"""

import numpy

import body
import tests


def deformation_gradient(deformed_basis, undeformed_basis):
    """Compute the deformation gradient tensor (F) from 2 basis vectors.

    :param frames.Basis deformed_basis: covariant basis object in the deformed configuration
    :param frames.Basis undeformed_basis: contravariant basis object in the undeformed configuration
    :return body.DeformationGradient
    """
    # Check that bases are compatible for matrix operations
    tests.check_covariance(deformed_basis, undeformed_basis)
    F = (numpy.outer(deformed_basis.vector1, undeformed_basis.vector1)
         + numpy.outer(deformed_basis.vector2, undeformed_basis.vector2)
         + numpy.outer(deformed_basis.vector3, undeformed_basis.vector3))
    result = body.DeformationGradient(F)
    return result


def green_lagrange_strain(right_cauchy_green_deformation_tensor):
    """Compute the Green-Lagrange strain tensor from the Left Cauchy-Green deformation tensor.

    :param numpy.ndarray right_cauchy_green_deformation_tensor: Left Cauchy-Green deformation tensor
    """
    result = 1 / 2 * (right_cauchy_green_deformation_tensor - numpy.identity(3))
    return result


def left_cauchy_green_deformation_tensor(deformation_gradient):
    """Compute the Left Cauchy-Green Deformation tensor (b) from the deformation gradient.

    :param numpy.ndarray deformation_gradient: Deformation gradient tensor
    """
    result = numpy.dot(deformation_gradient, deformation_gradient.T)
    return result


def right_cauchy_green_deformation_tensor(deformation_gradient):
    """Compute the Right Cauchy-Green Deformation tensor (C) from the deformation gradient.

    :param numpy.ndarray deformation_gradient: Deformation gradient tensor
    """
    result = numpy.dot(deformation_gradient.T, deformation_gradient)
    return result



