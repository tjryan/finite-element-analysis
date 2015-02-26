"""
errors.py contains all error definitions.

.. moduleauthor:: Tyler Ryan <tyler.ryan@engineering.ucla.edu>
"""


class BaseException(Exception):
    """Exception raised when simulation encounters a problem

    :param str message: description of error
    """

    def __init__(self, message):
        super(BaseException).__init__()
        self.message = message

    def __str__(self):
        return self.message


class BasisMismatchError(BaseException):
    """Attempted to perform matrix operations with two covariant or two contravariant vectors"""

    def __init__(self, basis1, basis2):
        super(BasisMismatchError, self).__init__(message='Bases are not compatible for matrix operations: '
                                                         + 'basis1 is type ' + basis1.type
                                                         + 'and basis2 is type ' + basis2.type)


class CompletenessError(BaseException):
    """Shape functions for given class do not satisfy completeness by interpolating a linear polynomial exactly."""

    def __init__(self, element_class, comparison_value, interpolated_value, error, tolerance):
        super(CompletenessError, self).__init__(
            message='Completeness is not satisfied for the element ' + element_class.__name__ + '. '
                    + 'The shape functions do not interpolate a random linear polynomial exactly. \n'
                    + 'comparison value: ' + str(comparison_value) + '\n'
                    + 'interpolated value: ' + str(interpolated_value) + '\n'
                    + 'error: ' + str(error) + '\n'
                    + 'tolerance: ' + str(tolerance))


class DifferentiationError(BaseException):
    """Computed derivative value in not within tolerance of the result from numerical differentiation"""

    def __init__(self, difference, tolerance):
        super(DifferentiationError, self).__init__(message='Computed derivative value is not within tolerance of '
                                                           + 'numerical differentiation result: \n'
                                                           + 'difference: ' + str(difference) + '\n'
                                                           + 'tolerance: ' + str(tolerance))


class InvalidArgumentError(BaseException):
    """Function passed an object of the wrong class"""

    def __init__(self, function, expected_class, actual_class):
        super(InvalidArgumentError, self).__init__(message='Function ' + function.__name__
                                                           + ' expected instance of ' + expected_class.__name__
                                                           + ' but was passed ' + actual_class.__name__)


class InvalidCoordinateError(BaseException):
    """The requested coordinate does not exist for the element."""

    def __init__(self, coordinate_index, coordinate_quantity):
        super(InvalidCoordinateError, self).__init__(
            message='The requested coordinate does not exist for this element. \n'
                    + 'coordinate index: ' + str(coordinate_index) + '\n'
                    + 'coordinate point_quantity: ' + str(coordinate_quantity))


class InvalidNodeError(BaseException):
    """The requested node does not exist for the element."""

    def __init__(self, node_index, node_quantity):
        super(InvalidNodeError, self).__init__(message='The requested node does not exist for this element. \n'
                                                       + 'node index: ' + str(node_index) + '\n'
                                                       + 'node point_quantity: ' + str(node_quantity))


class JacobianNegativeError(BaseException):
    """Jacobian has a negative value"""

    def __init__(self, jacobian):
        super(JacobianNegativeError, self).__init__(message='Jacobian has a negative value: J = ' + str(jacobian) + '.'
                                                            + ' Deformation gradient is not physical. The element is '
                                                            + 'likely too distorted.')


class MaterialFrameIndifferenceError(BaseException):
    """Material model is not frame indifferent."""

    def __init__(self, constitutive_model, quantity, difference, tolerance):
        super(MaterialFrameIndifferenceError, self).__init__(
            message='Constitutive model is not material frame indifferent.'
                    + ' Check equations for correctness: \n'
                    + 'Model: ' + constitutive_model.__class__.__name__ + '\n'
                    + 'Quantity: ' + quantity + ' changed by '
                    + str(difference) + ' compared to a tolerance of '
                    + str(tolerance) + '.')


class MaterialSymmetryError(BaseException):
    """Material model is not frame indifferent."""

    def __init__(self, constitutive_model, quantity, difference, tolerance):
        super(MaterialSymmetryError, self).__init__(
            message='Constitutive model does not satisfy material symmetry.'
                    + ' Check equations for correctness: \n'
                    + 'Model: ' + constitutive_model.__class__.__name__ + '\n'
                    + 'Quantity: ' + quantity + ' changed by '
                    + str(difference) + ' compared to a tolerance of '
                    + str(tolerance) + '.')


class NewtonMethodMaxIterationsExceededError(BaseException):
    """Newton's method solver has exceeded the max number of iterations without convergence."""

    def __init__(self, iterations, error, tolerance):
        super(NewtonMethodMaxIterationsExceededError, self).__init__(
            message='Newton\'s method solver has exceeded the max number of iterations without converging. \n'
                    + 'iterations: ' + str(iterations) + '\n'
                    + 'error: ' + str(error) + '\n'
                    + 'tolerance: ' + str(tolerance))


class PartitionNullityError(BaseException):
    """Partition of nullity is not satisfied for the shape functions of the given class."""

    def __init__(self, element_class, sum):
        super(PartitionNullityError, self).__init__(
            message='Partition of nullity is not satisfied for the element ' + element_class.__name__ + '. \n'
                    + 'sum: ' + str(sum) + '\n'
                    + 'required value: 0')


class PartitionUnityError(BaseException):
    """Partition of unity is not satisfied for the shape functions of the given class."""

    def __init__(self, element_class, sum):
        super(PartitionUnityError, self).__init__(
            message='Partition of unity is not satisfied for the element ' + element_class.__name__ + '. \n'
                    + 'sum: ' + str(sum) + '\n'
                    + 'required value: 1')


class PlaneStressError(BaseException):
    """Deformation gradient does not have the right structure for plane stress"""

    def __init__(self, deformation_gradient):
        super(PlaneStressError, self).__init__(message='Deformation gradient does not have the correct structure'
                                                       + ' for plane stress. It has non-zero shear components in the'
                                                       + ' normal direction: \n'
                                                       + 'F = ' + str(deformation_gradient))


class StretchRatioNegativeError(BaseException):
    """Stretch ratio assigned to a negative value in Newton's method solver."""

    def __init__(self, stretch_ratio):
        super(StretchRatioNegativeError, self).__init__(message='Stretch ratio assigned to a negative value in'
                                                                + ' Newton\'s method solver, likely due to a bad initial'
                                                                + ' guess. \n'
                                                                + 'stretch ratio: ' + str(stretch_ratio))

