"""
model_io_2.py module interfaces with the model and contains the contain that satisfies the requirements of assignment 2.

.. moduleauthor:: Tyler Ryan <tyler.ryan@engineering.ucla.edu>
"""
import constitutive_models
import elements
import materials
import model
import nodes
import operations
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


def test_element_calculations():
    """Test element creation and calculation of strain energy, force array, and stiffness matrix for a random
    deformation. This is a template for actual model implementation"""
    # TODO move some of this code into a model.run() method. This is a good template for implementation.
    # Create a new model
    fem = model.Model()
    # Choose a constitutive model for analysis
    fem.constitutive_model = constitutive_models.Neohookean
    # Choose a material for the body
    fem.material = materials.Custom('custom', 10, 5)
    # Choose a class of quadrature to use
    fem.quadrature_class = quadrature.GaussQuadratureOnePoint
    # fem.quadrature_class = quadrature.GaussQuadratureThreePoint
    # Choose an element type
    fem.element_type = elements.TriangularLinearElement
    # fem.element_type = elements.TriangularQuadraticElement
    # Choose number of degrees of freedom
    degrees_of_freedom = 2
    # Choose element thickness
    element_thickness = 1
    # Choose whether to enforce plane stress
    plane_stress = True
    # Create elements
    element_quantity = 1
    for i in range(element_quantity):
        # Create 3 nodes (specifically for the element)
        element_nodes = []
        for node_index in range(fem.element_type.node_quantity):
            if node_index < 3:
                node = nodes.CornerNode()
                element_nodes.append(node)
                # Add node to the model
                fem.nodes.append(node)
            else:
                node = nodes.MidpointNode()
                element_nodes.append(node)
                # Add node to the model
                fem.nodes.append(node)

        # Create nodes with random reference positions
        operations.generate_random_node_reference_positions(element_nodes)
        element = fem.element_type(constitutive_model=fem.constitutive_model,
                                   material=fem.material,
                                   quadrature_class=fem.quadrature_class,
                                   degrees_of_freedom=degrees_of_freedom,
                                   thickness=element_thickness,
                                   plane_stress=plane_stress)
        # Add the element to the model
        fem.elements.append(element)
        # Add element as parent element of nodes
        for node in element_nodes:
            node.parent_elements.append(element)
        # Assign nodes to element
        fem.assign_nodes()
        # Create element quadrature points
        element.create_quadrature_points()
    # Now that all elements have been created, update the current configuration of the model
    fem.update_current_configuration()
    # TODO verify correctness of internal forces and stiffness using 3 point numerical differentiation
    # TODO check the rank


def run():
    test_shape_functions()
    test_gauss_quadrature()
    test_element_calculations()


run()
