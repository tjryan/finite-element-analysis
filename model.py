"""
model.py module contains the primary components for constructing the finite element model.

.. moduleauthor:: Tyler Ryan <tyler.ryan@engineering.ucla.edu>
"""

import elements


class Model:
    """Finite Element Model containing globally needed values and functions for performing finite element analysis.

    :ivar material: material object that described the material the element is composed of
    :ivar constitutive_model: constitutive model class that describes the material behavior
    :ivar element_type: element class used as a template for all finite elements
    :ivar elements: list of element objects that make up the body
    """

    def __init__(self):
        self.material = None
        self.constitutive_model = None
        self.element_type = elements.TriangularLinearElement
        self.elements = []

        # LATER USE
        self.lab_frame = None  # LabFrame object
        self.reference_configuration = None  # ReferenceConfiguration object
        self.deformed_configuration = None  # DeformedConfiguration object


