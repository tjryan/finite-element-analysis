"""
constants.py contains constant values that are needed in the model.

.. moduleauthor:: Tyler Ryan <tyler.ryan@engineering.ucla.edu>
"""

FLOATING_POINT_TOLERANCE = 1e-12
"""The tolerated error for computations that should return exact matches, and are limited only by
16-digit floating point precision."""

NUMERICAL_DIFFERENTIATION_TOLERANCE = 1e-4
"""The tolerated error for numerical differentiation verification tests."""

NEWTON_METHOD_TOLERANCE = 1e-8
"""The tolerated error for convergence when solving using Newton's method."""
