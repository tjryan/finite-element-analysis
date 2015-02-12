"""
materials.py contains a library of materials and their properties to be used in the finite element model

All values are in SI units.

.. moduleauthor:: Tyler Ryan <tyler.ryan@engineering.ucla.edu>
"""


class Custom:
    """Custom material model for which the properties must be manually specified when initialized."""

    def __init__(self, name, first_lame_parameter, shear_modulus):
        self.name = name
        self.first_lame_parameter = first_lame_parameter
        self.shear_modulus = shear_modulus


class AluminumAlloy:
    """Material model for aluminum alloy"""

    name = 'aluminum alloy'
    first_lame_parameter = 53.5
    shear_modulus = 26.9


class Brass:
    """Material model for brass"""

    name = 'brass'
    first_lame_parameter = 72.3
    shear_modulus = 40.1


class Copper:
    """Material model for copper"""

    name = 'copper'
    first_lame_parameter = 87.6
    shear_modulus = 44.7


class Glass:
    """Material model for glass"""

    name = 'glass'
    first_lame_parameter = 17.4
    shear_modulus = 18.6


class Lead:
    """Material model for lead"""

    name = 'lead'
    first_lame_parameter = 48.2
    shear_modulus = 13.1


class TitaniumAlloy:
    """Material model for titanium alloy"""

    name = 'stainless steel'
    first_lame_parameter = 93.8
    shear_modulus = 42.4