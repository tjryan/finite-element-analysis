"""
operations.py contains useful function that are commonly used in the model but are not pre-defined in existing
packages such as numpy.

.. moduleauthor:: Tyler Ryan <tyler.ryan@engineering.ucla.edu>
"""
import random
import math

import numpy

import constants
import exceptions


def inverse_transpose(matrix):
    """
    Compute the inverse of the transpose of a matrix.

    :param numpy.ndarray matrix: matrix to be operated on
    :return: inverse-transpose of the matrix
    """
    return numpy.linalg.inv(matrix.T)


def calculate_lambda_from_E_and_G(E, G):
    """Calculate the value of the first lame parameter for a material, given values for Young's Modulus(E, in GPa)
    and the shear modulus (G, in GPa).

    :param float E: Young's Modulus of a material
    :param float G: Shear modulus of a material
    :return float first_lame_parameter: value for the first lame parameter (lambda)
    """
    first_lame_parameter = G * (E - 2 * G) / (3 * G - E)
    return first_lame_parameter


def generate_random_deformation_gradient(plane_stress=False):
    """Generate and return a random deformation gradient that is physically valid (Jacobian > 0)

    :param bool plane_stress: whether to restructure the matrix for plane stress
    :return numpy.ndarray random_deformation: a random 3x3 deformation gradient matrix
    """
    det = -1
    while det < 0:
        random_deformation = numpy.random.rand(3, 3)
        det = numpy.linalg.det(random_deformation)
    # If plane stress is requested, restructure the matrix for plane stress
    if plane_stress:
        random_deformation[0][2] = 0
        random_deformation[1][2] = 0
        random_deformation[2][0] = 0
        random_deformation[2][1] = 0
        random_deformation[2][2] = 1
    return random_deformation


def generate_random_rotation_matrix():
    """Generate and return a random rotation matrix.

    :return numpy.ndarray random_rotation: a random 3x3 rotation matrix.
    """
    # generate a random normalized vector
    random_vector = numpy.random.rand(3, 1)
    n = random_vector / numpy.linalg.norm(random_vector)
    # generate a random angle between 0 and pi
    angle = random.uniform(0, numpy.pi)
    # generate n_hat (see notes for structure)
    n_hat = numpy.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    # compute random rotation using the given equation
    rotation_matrix = (numpy.eye(3) - numpy.sin(angle) * n_hat
                       + (1 - numpy.cos(angle)) * (numpy.outer(n, n) - numpy.eye(3)))
    return rotation_matrix


def newton_method_thickness_stretch_ratio(material, constitutive_model, deformation_gradient, max_iterations=15):
    """Use Newton's method to iteratively solve for stretch ratio.

    :param material: material model for the body
    :param constitutive_model: constitutive model object described material behavior
    :param numpy.ndarray deformation_gradient: 3x3 deformation gradient matrix
    :param max_iterations: maximum number of iterations to perform while solving
    :return float thickness_stretch_ratio: ratio of the initial thickness to the deformed thickness
    """
    # TODO function to make a good initial guess
    # Make an initial guess for the thickness stretch ratio
    stretch_ratio = 1
    # Initialize the error and the current iteration counter
    error = float('inf')
    current_iteration = 0
    # Loop until the stress converges to within tolerance of zero, or max iterations is exceeded.
    while True:
        # Assign the 3-3 element of the deformation gradient to the current stretch ratio
        deformation_gradient[2][2] = stretch_ratio
        P33 = constitutive_model.first_piola_kirchhoff_stress(material=material,
                                                              deformation_gradient=deformation_gradient,
                                                              test=False)[2][2]
        error = math.fabs(0 - P33)
        # If the error is less than the tolerance, the loop has converged, so break out
        if error < constants.NEWTON_METHOD_TOLERANCE:
            break
        # Compute a new value for the stretch ratio and try again
        else:
            C3333 = constitutive_model.tangent_moduli(material=material,
                                                      deformation_gradient=deformation_gradient,
                                                      test=False)[2][2][2][2]
            # Compute correction to the stretch ratio
            delta_stretch = -(1 / C3333) * P33
            stretch_ratio += delta_stretch
            # If the loop has reached the max number of iterations, raise an error
            if current_iteration == max_iterations:
                raise exceptions.NewtonMethodMaxIterationsExceededError(iterations=max_iterations,
                                                                        error=error,
                                                                        tolerance=constants.NEWTON_METHOD_TOLERANCE)
            # Increment the iteration counter
            else:
                current_iteration += 1
    return stretch_ratio

