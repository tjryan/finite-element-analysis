"""
model.py module contains the primary components for constructing the finite element model.

.. moduleauthor:: Tyler Ryan <tyler.ryan@engineering.ucla.edu>
"""

import elements


class Model:
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


