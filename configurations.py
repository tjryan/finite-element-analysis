"""
frames.py contains the configurations that define the reference and deformation curvilinear coordinate systems.

.. moduleauthor:: Tyler Ryan <tyler.ryan@engineering.ucla.edu>
"""
import math

import numpy


class BaseConfiguration:
    """Base configuration that contains all the essential components to describe the curvilinear coordinate system
    for a quadrature point."""

    def __init__(self):
        self.midsurface_basis = [None] * 3
        self.midsurface_basis_contravariant = [None] * 3
        self.basis = [None] * 3
        self.basis_contravariant = [None] * 3
        self.midsurface_metric = None
        self.midsurface_metric_contravariant = None
        self.differential_area = None

    @staticmethod
    def compute_metric(basis):
        """Compute the elements of the symmetric covariant metric tensor and construct the matrix.

        :param list basis: list of three vectors that form a basis from which to compute the metric tensor
        """
        metric = numpy.zeros((len(basis), len(basis)))
        for i in range(len(basis)):
            for j in range(len(basis)):
                metric[i][j] = numpy.dot(basis[i], basis[j])
        return metric


class CurrentConfiguration(BaseConfiguration):
    """Current configuration for the quadrature point of an element, updates with every deformation."""

    def __init__(self):
        super(CurrentConfiguration, self).__init__()

    def update_configuration(self, element, quadrature_point):
        """Update all attributes of the reference configuration and assign them

        :param element: element containing the quadrature point
        :param quadrature_point: quadrature point object for which to create reference configuration
        """
        # Create in-plane midsurface basis vectors
        for coordinate_index in range(element.dimension):
            basis_vector = numpy.zeros(3)
            for lab_index in range(3):
                for node_index in range(element.node_quantity):
                    basis_vector[lab_index] += (
                        element.nodes[node_index].current_position[lab_index] * element.shape_function_derivatives(
                            node_index=node_index,
                            position=quadrature_point.position,
                            coordinate_index=coordinate_index))
            self.midsurface_basis[coordinate_index] = basis_vector
        # Compute the metric tensor
        self.midsurface_metric = self.compute_metric(basis=self.midsurface_basis[:2])
        # Compute the differential area
        self.differential_area = math.sqrt(numpy.linalg.det(self.midsurface_metric))
        # Compute transverse basis vector
        self.midsurface_basis[2] = numpy.cross(self.midsurface_basis[0],
                                               self.midsurface_basis[1]) / self.differential_area
        # Compute contravariant quantities
        self.midsurface_metric_contravariant = numpy.linalg.inv(self.midsurface_metric)
        for coordinate_index in range(element.dimension):
            self.midsurface_basis_contravariant[coordinate_index] = sum(
                [self.midsurface_metric_contravariant[coordinate_index][vector_index] * self.midsurface_basis[
                    vector_index] for
                 vector_index in range(element.dimension)])
        self.midsurface_basis_contravariant[2] = self.midsurface_basis[2]
        # Assign basis vectors from midsurface bases
        self.basis[0], self.basis[1] = self.midsurface_basis[:2]
        self.basis_contravariant[0], self.basis_contravariant[1] = self.midsurface_basis_contravariant[:2]
        # NOTE that we do not yet assign the transverse basis vectors because those are dependent on the stretch ratio
        # which has not yet been determined

    def update_transverse_basis_vectors(self, stretch_ratio):
        """Update the transverse basis vectors with the newly computed stretch ratio.

        :param float stretch_ratio: thickness stretch ratio computed from enforcing plane stress
        """
        basis_3 = self.midsurface_basis[2] * stretch_ratio
        basis_contravariant_3 = self.midsurface_basis_contravariant[2] / stretch_ratio
        self.basis[2] = basis_3
        self.basis_contravariant[2] = basis_contravariant_3


class ReferenceConfiguration(BaseConfiguration):
    """Reference configuration for the quadrature point of an element.

    :param element: element containing the quadrature point
    :param quadrature_point: quadrature point object for which to create reference configuration
    """

    def __init__(self, element, quadrature_point):
        super(ReferenceConfiguration, self).__init__()
        self.create_configuration(element, quadrature_point)

    def create_configuration(self, element, quadrature_point):
        """Compute all attributes of the reference configuration and assign them

        :param element: element containing the quadrature point
        :param quadrature_point: quadrature point object for which to create reference configuration
        """
        # Create in-plane midsurface basis vectors
        for coordinate_index in range(element.dimension):
            basis_vector = numpy.zeros(3)
            for lab_index in range(3):
                for node_index in range(element.node_quantity):
                    basis_vector[lab_index] += (
                        element.nodes[node_index].reference_position[lab_index] * element.shape_function_derivatives(
                            node_index=node_index,
                            position=quadrature_point.position,
                            coordinate_index=coordinate_index))
            self.midsurface_basis[coordinate_index] = basis_vector
        # Compute the metric tensor
        self.midsurface_metric = self.compute_metric(basis=self.midsurface_basis[:2])
        # Compute the differential area
        self.differential_area = math.sqrt(numpy.linalg.det(self.midsurface_metric))
        # Compute transverse basis vector
        self.midsurface_basis[2] = numpy.cross(self.midsurface_basis[0],
                                               self.midsurface_basis[1]) / self.differential_area
        # Compute contravariant quantities
        self.midsurface_metric_contravariant = numpy.linalg.inv(self.midsurface_metric)
        for coordinate_index in range(element.dimension):
            self.midsurface_basis_contravariant[coordinate_index] = sum(
                [self.midsurface_metric_contravariant[coordinate_index][vector_index] * self.midsurface_basis[
                    vector_index] for
                 vector_index in range(element.dimension)])
        self.midsurface_basis_contravariant[2] = self.midsurface_basis[2]
        # Assign basis vectors from midsurface bases
        self.basis = self.midsurface_basis
        self.basis_contravariant = self.midsurface_basis_contravariant

