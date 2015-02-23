"""
tests.py module contains all verification tests used to ensure correctness of the code.

.. moduleauthor:: Tyler Ryan <tyler.ryan@engineering.ucla.edu>
"""

import math

import numpy

import constants
import constitutive_models
import exceptions
import materials
import operations


def check_covariance(basis1, basis2):
    """Check that basis1 and basis 2 are compatible for operations, namely that
    one of them is covariant and the other is contravariant.
    """
    types = [basis1.type, basis2.type]
    # If there is not one covariant and one contravariant basis, raise an error
    if types.count(constants.COVARIANT) != 1 or types.count(constants.CONTRAVARIANT) != 1:
        raise exceptions.BasisMismatchError(basis1, basis2)


def check_deformation_gradient_physical(jacobian):
    """Check that the deformation gradient makes physical sense by checking that the Jacobian is positive.

    :param float jacobian: jacobian to check
    """
    if jacobian <= 0:
        raise exceptions.JacobianNegativeError(jacobian=jacobian)


def check_deformation_gradient_plane_stress(deformation_gradient):
    """Check that the deformation gradient has the correct structure for a plane stress case."""
    if (deformation_gradient[0][2] != 0 or deformation_gradient[1][2] != 0
        or deformation_gradient[2][0] != 0 or deformation_gradient[2][1] != 0):
        raise exceptions.PlaneStressError(deformation_gradient=deformation_gradient)


def material_frame_indifference():
    """Check for frame indifference of the material model by verifying that the strain energy density, first
    Piola-Kirchhoff stress, and tangent_moduli all remain unchanged under a random rotation.
    """
    # Generate a random deformation gradient
    random_deformation = operations.generate_random_deformation_gradient()
    # Choose a material and constitutive model to test
    material = materials.AluminumAlloy()
    constitutive_model = constitutive_models.Neohookean()
    # Compute quantities for the material model
    w = constitutive_model.strain_energy_density(material=material, deformation_gradient=random_deformation)
    p = constitutive_model.first_piola_kirchhoff_stress(material=material, deformation_gradient=random_deformation,
                                                        test=True)
    c = constitutive_model.tangent_moduli(material=material, deformation_gradient=random_deformation, test=True)
    # Generate a random rotation matrix (call a separate function from operations.py)
    random_rotation = operations.generate_random_rotation_matrix()
    rotated_deformation = numpy.dot(random_rotation, random_deformation)
    # Compute quantities for the rotated deformation gradient
    w_rotated = constitutive_model.strain_energy_density(material=material, deformation_gradient=rotated_deformation)
    p_rotated = constitutive_model.first_piola_kirchhoff_stress(material=material,
                                                                deformation_gradient=rotated_deformation, test=True)
    c_rotated = constitutive_model.tangent_moduli(material=material, deformation_gradient=rotated_deformation,
                                                  test=True)
    # Test that each element is within tolerance of its original value
    w_error = math.fabs(w - w_rotated)
    if w_error > constants.ERROR_TOLERANCE:
        raise exceptions.MaterialFrameIndifferenceError(constitutive_model=constitutive_model,
                                                        quantity='strain energy density',
                                                        difference=w_error,
                                                        tolerance=constants.ERROR_TOLERANCE)
    p_errors = []
    p_comparison = numpy.dot(random_rotation, p)
    for i in range(3):
        for j in range(3):
            p_error = math.fabs(p_rotated[i][j] - p_comparison[i][j])
            p_errors.append(p_error)
    p_max_error = max(p_errors)
    if p_max_error > constants.ERROR_TOLERANCE:
        raise exceptions.MaterialFrameIndifferenceError(constitutive_model=constitutive_model,
                                                        quantity='first Piola-Kirchhoff stress',
                                                        difference=p_max_error,
                                                        tolerance=constants.ERROR_TOLERANCE)
    c_errors = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    c_comparison = 0
                    for m in range(3):
                        for n in range(3):
                            c_comparison += random_rotation[i][m] * random_rotation[k][n] * c[m][j][n][l]
                    c_error = math.fabs(c_rotated[i][j][k][l] - c_comparison)
                    c_errors.append(c_error)
    c_max_error = max(c_errors)
    if c_max_error > constants.ERROR_TOLERANCE:
        raise exceptions.MaterialFrameIndifferenceError(constitutive_model=constitutive_model,
                                                        quantity='tangent moduli',
                                                        difference=c_max_error,
                                                        tolerance=constants.ERROR_TOLERANCE)


def material_symmetry():
    """Check for symmetries of the material model by verifying that the strain energy density remains invariant
    under rotation, and that the first Piola-Kirchhoff stress and tangent moduli transform appropriately.
    """
    # Generate a random deformation gradient
    random_deformation = operations.generate_random_deformation_gradient()
    # Choose a material and constitutive model to test
    material = materials.AluminumAlloy
    constitutive_model = constitutive_models.Neohookean()
    # Compute quantities for the material model
    w = constitutive_model.strain_energy_density(material=material, deformation_gradient=random_deformation)
    p = constitutive_model.first_piola_kirchhoff_stress(material=material, deformation_gradient=random_deformation,
                                                        test=True)
    c = constitutive_model.tangent_moduli(material=material, deformation_gradient=random_deformation, test=True)
    # Generate a random rotation matrix (call a separate function from operations.py)
    random_rotation = operations.generate_random_rotation_matrix()
    rotated_deformation = numpy.dot(random_deformation, random_rotation)
    # Compute quantities for the rotated deformation gradient
    w_rotated = constitutive_model.strain_energy_density(material=material, deformation_gradient=rotated_deformation)
    p_rotated = constitutive_model.first_piola_kirchhoff_stress(material=material,
                                                                deformation_gradient=rotated_deformation, test=True)
    c_rotated = constitutive_model.tangent_moduli(material=material, deformation_gradient=rotated_deformation,
                                                  test=True)
    # Test that each element is within tolerance of its original value
    w_error = math.fabs(w - w_rotated)
    if w_error > constants.ERROR_TOLERANCE:
        raise exceptions.MaterialSymmetryError(constitutive_model=constitutive_model,
                                               quantity='strain energy density',
                                               difference=w_error,
                                               tolerance=constants.ERROR_TOLERANCE)
    p_errors = []
    p_comparison = numpy.dot(random_rotation, p)
    for i in range(3):
        for j in range(3):
            p_comparison = 0
            for k in range(3):
                p_comparison += random_rotation[k][j] * p[i][k]
            p_error = math.fabs(p_rotated[i][j] - p_comparison)
            p_errors.append(p_error)
    p_max_error = max(p_errors)
    if p_max_error > constants.ERROR_TOLERANCE:
        raise exceptions.MaterialSymmetryError(constitutive_model=constitutive_model,
                                               quantity='first Piola-Kirchhoff stress',
                                               difference=p_max_error,
                                               tolerance=constants.ERROR_TOLERANCE)
    c_errors = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    c_comparison = 0
                    for m in range(3):
                        for n in range(3):
                            c_comparison += random_rotation[m][j] * random_rotation[n][l] * c[i][m][k][n]
                    c_error = math.fabs(c_rotated[i][j][k][l] - c_comparison)
                    c_errors.append(c_error)
    c_max_error = max(c_errors)
    if c_max_error > constants.ERROR_TOLERANCE:
        raise exceptions.MaterialSymmetryError(constitutive_model=constitutive_model,
                                               quantity='tangent moduli',
                                               difference=c_max_error,
                                               tolerance=constants.ERROR_TOLERANCE)


def verify_first_piola_kirchhoff_stress(constitutive_model, material, deformation_gradient,
                                        first_piola_kirchhoff_stress, h=1e-6):
    """Verify that tensor is within tolerance of the result from numerical integration using the 3 point method.

    :param constitutive_model: a constitutive model object from constitutive_models.py
    :param material: the material undergoing deformation
    :param deformation_gradient: deformation gradient matrix used to compute the Piola-Kirchhoff Stress
    :param first_piola_kirchhoff_stress: the stress tensor calculated from the deformation gradient
    :param float h: a small deviation that perturbs the evaluation points of the stress tensor
    """
    errors = []
    # For each element in the stress tensor
    for i in range(3):
        for j in range(3):
            # make a copy of the deformation gradient to perturb
            perturbed_deformation = deformation_gradient.copy()
            # Perturb the element in the positive direction
            perturbed_deformation[i][j] += h
            # Compute positively perturbed strain energy density
            strain_energy_density_plus = constitutive_model.strain_energy_density(material=material,
                                                                                  deformation_gradient=perturbed_deformation)
            # Perturb the element in the negative direction
            perturbed_deformation[i][j] -= 2 * h
            # Compute the negatively perturbed strain energy density
            strain_energy_density_minus = constitutive_model.strain_energy_density(material=material,
                                                                                   deformation_gradient=perturbed_deformation)
            # Compute the result of numerical differentiation
            numerical_value = (strain_energy_density_plus - strain_energy_density_minus) / (2 * h)
            computed_value = first_piola_kirchhoff_stress[i][j]
            error = math.fabs(computed_value - numerical_value)
            errors.append(error)
    max_error = max(errors)
    # If the result is not within tolerance of the provided value, raise an error
    if max_error > constants.ERROR_TOLERANCE:
        raise exceptions.DifferentiationError(difference=max_error,
                                              tolerance=constants.ERROR_TOLERANCE)


def verify_tangent_moduli(constitutive_model, material, deformation_gradient, tangent_moduli, h=1e-6):
    """Verify that the computed tangent moduli tensor is within tolerance of the result from numerical differentiation
    using the 3 point method.

    :param constitutive_model: a constitutive model object from constitutive_models.py
    :param material: the material undergoing deformation
    :param deformation_gradient: deformation gradient matrix used to compute the tangent moduli
    :param tangent_moduli: tangent moduli tensor computed from the provided deformation gradient
    :param float h: a small deviation that perturbs the evaluation points of the stress tensor
    """
    errors = []
    # For each element of the tangent moduli
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    # Make a copy of the deformation gradient to perturb
                    perturbed_deformation = deformation_gradient.copy()
                    # Perturb in the positive direction
                    perturbed_deformation[k][l] += h
                    piola_kirchhoff_plus = constitutive_model.first_piola_kirchhoff_stress(material=material,
                                                                                           deformation_gradient=perturbed_deformation,
                                                                                           test=False)
                    # Perturb in the negative direction
                    perturbed_deformation[k][l] -= 2 * h
                    piola_kirchhoff_minus = constitutive_model.first_piola_kirchhoff_stress(material=material,
                                                                                            deformation_gradient=perturbed_deformation,
                                                                                            test=False)
                    # Compute the result of numerical differentiation
                    numerical_value = (piola_kirchhoff_plus[i][j] - piola_kirchhoff_minus[i][j]) / (2 * h)
                    computed_value = tangent_moduli[i][j][k][l]
                    error = math.fabs(computed_value - numerical_value)
                    errors.append(error)
    max_error = max(errors)
    # If the result is not within tolerance of the provided value, raise an error
    if max_error > constants.ERROR_TOLERANCE:
        raise exceptions.DifferentiationError(difference=max_error,
                                              tolerance=constants.ERROR_TOLERANCE)



