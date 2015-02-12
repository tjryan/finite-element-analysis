"""
frames.py contains the coordinate systems and mappings used to describe the body as it deforms.

.. moduleauthor:: Tyler Ryan <tyler.ryan@engineering.ucla.edu>
"""

import numpy

import constants


class Basis:
    """A set of three covariant or contravariant basis vectors that are used to fully describe points in space"""

    def __init__(self, vector1, vector2, vector3, type):
        self.vector1 = vector1
        self.vector2 = vector2
        self.vector3 = vector3
        self.type = type
        self.metric_tensor = self.compute_metric_tensor()

    def compute_metric_tensor(self):
        """Compute the elements of the symmetric metric tensor and construct the matrix."""
        g11 = numpy.dot(self.vector1, self.vector1)
        g12 = numpy.dot(self.vector1, self.vector2)
        g13 = numpy.dot(self.vector1, self.vector3)
        g22 = numpy.dot(self.vector2, self.vector2)
        g23 = numpy.dot(self.vector2, self.vector3)
        g33 = numpy.dot(self.vector3, self.vector3)
        covariant_metric_tensor = numpy.matrix([[g11, g12, g13],
                                                [g12, g22, g23],
                                                [g13, g23, g33]])
        if self.type == constants.COVARIANT:
            return covariant_metric_tensor
        # If the type is contravariant, return the inverse metric tensor
        else:
            contravariant_metric_tensor = numpy.linalg.inv(covariant_metric_tensor)
            return contravariant_metric_tensor


class DeformedConfiguration:
    """Deformed, current configuration of the body described in the same curvilinear coordinate system used
    by the lab frame.
    """

    def __init__(self, covariant_basis, contravariant_basis):
        self.covariant_basis = None
        self.contravariant_basis = None
        self.mapping = None


class LabFrame:
    """Lab frame, described by a convenient curvilinear coordinate system based on the geometry
    of the body that maps to Cartesian coordinates.
    """

    def __init__(self, basis):
        self.basis = None
        self.mapping = None


class ReferenceConfiguration:
    """Undeformed, initial configuration of the body described in the same curvilinear coordinate system
    used by the lab frame.
    """

    def __init__(self, covariant_basis, contravariant_basis):
        self.covariant_basis = None
        self.contravariant_basis = None
        self.mapping = None
