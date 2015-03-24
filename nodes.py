"""
nodes.py contains the nodes for the finite elements.

.. moduleauthor:: Tyler Ryan <tyler.ryan@engineering.ucla.edu>
"""
import numpy


class Node:
    """Point in space that defines the boundaries of an element and the connection points between elements.
    Also defines where the degrees of freedom are defined, such as displacements and rotations due to loading.

    :param int global_id: unique global ID for the node
    :param list reference_position: reference position of the node in 3D
    :param list prescribed_displacements: prescribed displacements for each degree of freedom of the node
    """

    def __init__(self, global_id, reference_position, prescribed_displacements):
        self.global_id = global_id
        self.reference_position = reference_position
        self.current_position = reference_position.copy()
        self.prescribed_displacements = prescribed_displacements

        # For displacement solving
        self.displacement_step = numpy.array([None] * 3)


class CornerNode(Node):
    """A node located at the corner of an element.

    :param int global_id: unique global ID for the node
    :param list reference_position: reference position of the node in 3D
    :param list prescribed_displacements: prescribed displacements for each degree of freedom of the node
    """

    def __init__(self, global_id, reference_position, prescribed_displacements):
        super(CornerNode, self).__init__(global_id=global_id,
                                         reference_position=reference_position,
                                         prescribed_displacements=prescribed_displacements)


class MidpointNode(Node):
    """A node located at the midpoint of an element face.

    :param int global_id: unique global ID for the node
    :param list reference_position: reference position of the node in 3D
    :param list prescribed_displacements: prescribed displacements for each degree of freedom of the node
    """

    def __init__(self, global_id, reference_position, prescribed_displacements):
        super(MidpointNode, self).__init__(global_id=global_id,
                                           reference_position=reference_position,
                                           prescribed_displacements=prescribed_displacements)
