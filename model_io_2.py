"""
model_io_2.py module interfaces with the model and contains the contain that satisfies the requirements of assignment 2.

.. moduleauthor:: Tyler Ryan <tyler.ryan@engineering.ucla.edu>
"""
import elements
import quadrature
import tests


def test_shape_functions():
    """Run the verification tests to check that the shape functions for the 3 and 6 node triangular elements
    are implemented correctly."""
    # Test 3 node element
    triangular_linear_element = elements.TriangularLinearElement
    tests.shape_functions(element_class=triangular_linear_element)
    # Test 6 node element
    triangular_quadratic_element = elements.TriangularQuadraticElement
    tests.shape_functions(element_class=triangular_quadratic_element)


def test_gauss_quadrature():
    """Run verification test to check that the result of numerical integration using Gauss quadrature matches
    the result of exact integration for first and second order polynomials for an isoparametric triangular element."""
    # Test one point quadrature
    one_point_quadrature = quadrature.GaussQuadratureOnePoint
    tests.gauss_quadrature(quadrature_class=one_point_quadrature)
    # Test three point quadrature
    three_point_quadrature = quadrature.GaussQuadratureThreePoint
    tests.gauss_quadrature(quadrature_class=three_point_quadrature)


def run():
    test_shape_functions()
    test_gauss_quadrature()


run()
