"""
constants.py contains constant values that are needed in the model

.. moduleauthor:: Tyler Ryan <tyler.ryan@engineering.ucla.edu>
"""

COVARIANT = 'covariant'
"""Indicates that a Basis is covariant and expressed in curvilinear coordinates."""

CONTRAVARIANT = 'contravariant'
"""Indicates that a Basis is contravariant and expressed in curvilinear coordinates."""

LAB = 'lab'
"""Indicates that a Basis is in the lab frame and is therefore expresses in Cartesian coordinates."""

ERROR_TOLERANCE = 1e-2
"""The tolerated error for verification tests."""

NEWTON_METHOD_TOLERANCE = 1e-6
"""The tolerated error for convergence when solving using Newton's method."""
