"""
model.py module contains the primary components for constructing the finite element model.

.. moduleauthor:: Tyler Ryan <tyler.ryan@engineering.ucla.edu>
"""

import elements


class Model:
    """Finite Element Model containing globally needed values and functions for performing finite element analysis.

    :ivar material: material object that described the material the element is composed of
    :ivar constitutive_model: constitutive model class that describes the material behavior
    :ivar quadrature_class: quadrature class to use for elements
    :ivar element_type: element class used as a template for all finite elements
    :ivar list nodes: list of all node objects in the mesh
    :ivar list elements: list of element objects that make up the body
    """
    # TODO add arguments to init that are necessary for every analysis
    def __init__(self):
        self.material = None
        self.constitutive_model = None
        self.quadrature_class = None
        self.element_type = elements.TriangularLinearElement
        self.nodes = []
        self.elements = []

        # LATER USE
        self.lab_frame = None  # LabFrame object
        self.reference_configuration = None  # ReferenceConfiguration object
        self.deformed_configuration = None  # DeformedConfiguration object

    def assign_nodes(self):
        """Add nodes to their parent elements."""
        for node in self.nodes:
            for parent_element in node.parent_elements:
                parent_element.nodes.append(node)

    def create_elements(self):
        """Create elements."""
        pass

    def update_current_configuration(self):
        """Update the current configuration of all elements in the model."""
        for element in self.elements:
            element.update_current_configuration()
