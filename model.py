"""
model.py module contains the primary components for constructing the finite element model.

.. moduleauthor:: Tyler Ryan <tyler.ryan@engineering.ucla.edu>
"""
import elements


class Element:
    """An element of a mesh that discretizes the domain of the body. Contains information about the behavior
    of nodes and quadrature points, as well as material properties.
    """

    def __init__(self):
        self.nodes = []  # list of node objects
        self.quadrature_points = []  # list of quadrature point objects

    def internal_nodal_force_array(self):
        """Computes the 2D internal nodal force array for the element."""
        pass

    def jacobian_matrix(self):
        """Computes the Jacobian matrix for the element."""
        pass

    def stiffness_tensor(self):
        """Computes the 4-D stiffness tensor for the element."""
        pass

    def strain_energy(self):
        """Computes the total strain energy of element."""
        pass


class FEM:
    """Finite Element Model containing globally needed values and functions for performing finite element analysis."""

    def __init__(self):
        self.constitutive_model = None  # class from constitutive_models.py
        self.material = None  # Material object
        self.element = elements.TriangularLinearElement  # Template element object
        self.elements = []  # list of Element objects


        # LATER USE
        self.lab_frame = None  # LabFrame object
        self.reference_configuration = None  # ReferenceConfiguration object
        self.deformed_configuration = None  # DeformedConfiguration object


class Node:
    """Point in space that defines the boundaries of an element and the connection points between elements.
    Also defines where the degrees of freedom are defined, such as displacements and rotations due to loading.
    """

    def __init__(self):
        self.id = None  # identifier for the node
        self.reference_position = None  # vector
        self.current_position = None  # vector
        self.parent_elements = []  # the elements object that this node belongs to


class QuadraturePoint:
    """
    A point in an element at which quantities of interest can be easily evaluated using Gauss quadrature.
    Stores values of shape functions, kinetic, and kinematic quantities at this point.
    """
    # TODO rethink the use of this class. It might not be needed anymore.

    def __init__(self):
        self.position = None
        self.weight = None
        self.remainder = None
        self.deformation_gradient = None
        self.first_piola_kirchhoff_stress = None
        self.tangent_moduli = None
        self.strain_energy_density = None


class ShapeFunction:
    """Function that interpolates the solution between discrete values obtained at the nodes."""

    def __init__(self, element):
        if isinstance(element, elements.TriangularLinearElement):
            pass
