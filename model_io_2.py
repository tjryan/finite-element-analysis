"""
model_io_2.py module interfaces with the model and contains the contain that satisfies the requirements of assignment 2.

.. moduleauthor:: Tyler Ryan <tyler.ryan@engineering.ucla.edu>
"""
import elements
import tests


def test_shape_functions():
    """Run the verification tests to check that the shape functions for the 3 and 6 node triangular elements
    are implemented correctly."""
    # Test 3 node element
    triangular_element_three_node = elements.TriangularElementThreeNode
    tests.shape_functions_triangular_element(triangular_element_three_node)
    # Test 6 node element
    triangular_element_six_node = elements.TriangularElementSixNode
    tests.shape_functions_triangular_element(triangular_element_six_node)


def run():
    test_shape_functions()


run()
