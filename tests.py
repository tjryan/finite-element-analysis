"""
tests.py module contains all verification tests used to ensure correctness of the code.

.. moduleauthor:: Tyler Ryan <tyler.ryan@engineering.ucla.edu>
"""

import math
import random

import numpy
from scipy.integrate import dblquad

import constants
import constitutive_models
import exceptions
import materials
import operations


def covariance(basis1, basis2):
    """Check that basis1 and basis 2 are compatible for operations, namely that
    one of them is covariant and the other is contravariant.
    """
    types = [basis1.type, basis2.type]
    # If there is not one covariant and one contravariant basis, raise an error
    if types.count(constants.COVARIANT) != 1 or types.count(constants.CONTRAVARIANT) != 1:
        raise exceptions.BasisMismatchError(basis1, basis2)


def deformation_gradient_physical(jacobian):
    """Check that the deformation gradient makes physical sense by checking that the Jacobian is positive.

    :param float jacobian: jacobian to check
    """
    if jacobian <= 0:
        raise exceptions.JacobianNegativeError(jacobian=jacobian)


def deformation_gradient_plane_stress(deformation_gradient):
    """Check that the deformation gradient has the correct structure for a plane stress case."""
    if (deformation_gradient[0][2] != 0 or deformation_gradient[1][2] != 0
        or deformation_gradient[2][0] != 0 or deformation_gradient[2][1] != 0):
        raise exceptions.PlaneStressError(deformation_gradient=deformation_gradient)


def material_frame_indifference():
    """Check for frame indifference of the material model by verifying that the strain energy density, first
    Piola-Kirchhoff stress, and tangent_moduli_numerical_differentiation all remain unchanged under a random rotation.
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
    if w_error > constants.FLOATING_POINT_TOLERANCE:
        raise exceptions.MaterialFrameIndifferenceError(constitutive_model=constitutive_model,
                                                        quantity='strain energy density',
                                                        difference=w_error,
                                                        tolerance=constants.FLOATING_POINT_TOLERANCE)
    p_errors = []
    p_comparison = numpy.dot(random_rotation, p)
    for i in range(3):
        for j in range(3):
            p_error = math.fabs(p_rotated[i][j] - p_comparison[i][j])
            p_errors.append(p_error)
    p_max_error = max(p_errors)
    if p_max_error > constants.FLOATING_POINT_TOLERANCE:
        raise exceptions.MaterialFrameIndifferenceError(constitutive_model=constitutive_model,
                                                        quantity='first Piola-Kirchhoff stress',
                                                        difference=p_max_error,
                                                        tolerance=constants.FLOATING_POINT_TOLERANCE)
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
    if c_max_error > constants.FLOATING_POINT_TOLERANCE:
        raise exceptions.MaterialFrameIndifferenceError(constitutive_model=constitutive_model,
                                                        quantity='tangent moduli',
                                                        difference=c_max_error,
                                                        tolerance=constants.FLOATING_POINT_TOLERANCE)


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
    if w_error > constants.FLOATING_POINT_TOLERANCE:
        raise exceptions.MaterialSymmetryError(constitutive_model=constitutive_model,
                                               quantity='strain energy density',
                                               difference=w_error,
                                               tolerance=constants.FLOATING_POINT_TOLERANCE)
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
    if p_max_error > constants.FLOATING_POINT_TOLERANCE:
        raise exceptions.MaterialSymmetryError(constitutive_model=constitutive_model,
                                               quantity='first Piola-Kirchhoff stress',
                                               difference=p_max_error,
                                               tolerance=constants.FLOATING_POINT_TOLERANCE)
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
    if c_max_error > constants.FLOATING_POINT_TOLERANCE:
        raise exceptions.MaterialSymmetryError(constitutive_model=constitutive_model,
                                               quantity='tangent moduli',
                                               difference=c_max_error,
                                               tolerance=constants.FLOATING_POINT_TOLERANCE)


def first_piola_kirchhoff_stress_numerical_differentiation(constitutive_model, material, deformation_gradient,
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
    if max_error > constants.NUMERICAL_DIFFERENTIATION_TOLERANCE:
        raise exceptions.DifferentiationError(difference=max_error,
                                              tolerance=constants.NUMERICAL_DIFFERENTIATION_TOLERANCE)


def gauss_quadrature(quadrature_class):
    """Check numerical integration using Gauss quadrature against exact integration for an isoparametric
    triangular element for first and second order polynomials."""
    # For 1st and 2nd order polynomials
    for order in [1, 2]:
        random_polynomial = None
        if order == 1:
            # Define a random linear polynomial
            random_coefficient_1 = random.uniform(-1, 1)
            random_coefficient_2 = random.uniform(-1, 1)
            random_polynomial = lambda r, s: random_coefficient_1 * r + random_coefficient_2 * s
        if order == 2:
            # Define a random quadratic polynomial
            random_coefficient_1 = random.uniform(-1, 1)
            random_coefficient_2 = random.uniform(-1, 1)
            random_coefficient_3 = random.uniform(-1, 1)
            random_coefficient_4 = random.uniform(-1, 1)
            random_coefficient_5 = random.uniform(-1, 1)
            random_polynomial = (lambda r, s: random_coefficient_1 * r + random_coefficient_2 * s
                                              + random_coefficient_3 * r * s + random_coefficient_4 * r ** 2
                                              + random_coefficient_5 * s ** 2)
        # Compare the result of numerical integration using Gauss quadrature against exact integration
        gauss_integration_value = quadrature_class.integrate(function=random_polynomial)
        s_min = 0
        s_max = 1
        r_min = lambda s: 0
        r_max = lambda s: 1 - s
        exact_integration_value, exact_integration_error = dblquad(random_polynomial, s_min, s_max, r_min, r_max)
        error = math.fabs(gauss_integration_value - exact_integration_value)


def shape_functions_completeness(element_class):
    """Check shape functions of the given element for completeness by testing if they can interpolate a
    random linear polynomial exactly.

    :param element_class: class of element to test
    """
    # Define a random linear polynomial
    random_coefficient_1 = random.uniform(-1, 1)
    random_coefficient_2 = random.uniform(-1, 1)
    random_linear_polynomial = lambda r, s: random_coefficient_1 * r + random_coefficient_2 * s
    # Generate random point
    random_r = random.uniform(0, 1)
    random_s = random.uniform(0, random_r)
    # Evaluate the polynomial at this point
    comparison_value = random_linear_polynomial(random_r, random_s)
    # Sample the polynomial at the nodes
    node_values = [random_linear_polynomial(node_position[0], node_position[1]) for node_position in
                   element_class.node_positions]
    # Interpolate sample node values using the shape functions to compute that value at a the random point
    # by summing over nodes
    interpolated_value = sum(
        [node_values[node_index] * element_class.shape_functions(node_index, r=random_r, s=random_s) for node_index in
         range(element_class.node_quantity)])
    # Compute the error between the comparison and interpolated values, and raise an error if not within tolerance
    # of each other
    error = math.fabs(interpolated_value - comparison_value)
    if error > constants.FLOATING_POINT_TOLERANCE:
        raise exceptions.CompletenessError(element_class=element_class, comparison_value=comparison_value,
                                           interpolated_value=interpolated_value, error=error,
                                           tolerance=constants.FLOATING_POINT_TOLERANCE)


def shape_functions_partition_nullity(element_class, r, s):
    """Check that the shape functions for the given element class verify partition of nullity at the specified
    coordinates.

    Partition of nullity: the sum of the derivatives of the shape functions at any point r, s should equal 0.

    :param element_class: class of element to test
    :param float r: random coordinate
    :param float s: random coordinate
    """
    partition_nullity_sum = 0
    for coordinate_index in range(element_class.dimension):
        partition_nullity_sum += sum([
            element_class.shape_function_derivatives(node_index=node_index, r=r, s=s,
                                                     coordinate_index=coordinate_index) for node_index in
            range(element_class.node_quantity)])
    # Compute the error with the expected value of 0
    partition_nullity_error = math.fabs(partition_nullity_sum - 0)
    # If the error is not within tolerance of the expected value, raise an error
    if partition_nullity_error > constants.FLOATING_POINT_TOLERANCE:
        raise exceptions.PartitionNullityError(element_class=element_class, sum=partition_nullity_sum)


def shape_functions_partition_unity(element_class, r, s):
    """Check that the shape functions for the given element class verify partition of unity at the specified
    coordinates.

    Partition of unity: the sum of shape function at any point r, s should equal 1.

    :param element_class: class of element to test
    :param float r: random coordinate
    :param float s: random coordinate
    """
    partition_unity_sum = sum(
        [element_class.shape_functions(node_index=node_index, r=r, s=s) for node_index in range(
            element_class.node_quantity)])
    # Compute the error from the expected value of 1
    partition_unity_error = math.fabs(partition_unity_sum - 1)
    # If the error is not within tolerance of the expected value, raise an error
    if partition_unity_error > constants.FLOATING_POINT_TOLERANCE:
        raise exceptions.PartitionUnityError(element_class=element_class, sum=partition_unity_sum)


def shape_functions_triangular_element(element_class):
    """Verify that the shape functions for the given triangular element class are implemented correctly by checking that
    they satisfy partition of unity and that the derivatives satisfy partition of nullity.

    :param element_class: class of element to test
    """
    # Generate random coordinates
    random_r = random.uniform(0, 1)
    random_s = random.uniform(0, random_r)
    # Perform checks
    shape_functions_partition_unity(element_class=element_class, r=random_r, s=random_s)
    shape_functions_partition_nullity(element_class=element_class, r=random_r, s=random_s)
    shape_functions_numerical_differentiation(element_class=element_class, r=random_r, s=random_s)
    shape_functions_completeness(element_class=element_class)


def shape_functions_numerical_differentiation(element_class, r, s, h=1e-6):
    """Verify that tensor is within tolerance of the result from numerical integration using the 3 point method.

    :param element_class: class of element to test
    :param float h: a small deviation that perturbs the evaluation points of the stress tensor
    """
    for node_index in range(element_class.node_quantity):
        for coordinate_index in range(element_class.dimension):
            # Check shape function derivative with respect to the specified coordinate by perturbing it
            perturbed_r_plus = r + h * (coordinate_index == 0)
            perturbed_s_plus = s + h * (coordinate_index == 1)
            perturbed_r_minus = r - h * (coordinate_index == 0)
            perturbed_s_minus = s - h * (coordinate_index == 1)
            shape_function_plus = element_class.shape_functions(node_index=node_index, r=perturbed_r_plus,
                                                                s=perturbed_s_plus)
            shape_function_minus = element_class.shape_functions(node_index=node_index, r=perturbed_r_minus,
                                                                 s=perturbed_s_minus)
            numerical_value = (shape_function_plus - shape_function_minus) / (2 * h)
            computed_value = element_class.shape_function_derivatives(node_index=node_index, r=r, s=s,
                                                                      coordinate_index=coordinate_index)
            error = math.fabs(computed_value - numerical_value)
            # If the result is not within tolerance of the provided value, raise an error
            if error > constants.NUMERICAL_DIFFERENTIATION_TOLERANCE:
                raise exceptions.DifferentiationError(difference=error,
                                                      tolerance=constants.NUMERICAL_DIFFERENTIATION_TOLERANCE)


def tangent_moduli_numerical_differentiation(constitutive_model, material, deformation_gradient, tangent_moduli,
                                             h=1e-6):
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
    if max_error > constants.NUMERICAL_DIFFERENTIATION_TOLERANCE:
        raise exceptions.DifferentiationError(difference=max_error,
                                              tolerance=constants.NUMERICAL_DIFFERENTIATION_TOLERANCE)



