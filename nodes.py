"""
nodes.py contains the nodes for the finite elements.

.. moduleauthor:: Tyler Ryan <tyler.ryan@engineering.ucla.edu>
"""
import operations


class Node:
    """Point in space that defines the boundaries of an element and the connection points between elements.
    Also defines where the degrees of freedom are defined, such as displacements and rotations due to loading.

    :param int global_id: unique global ID for the node
    :param list reference_position: reference position of the node in 3D
    :param list prescribed_dof: prescribed displacements for each degree of freedom of the node
    """

    def __init__(self, global_id, reference_position, prescribed_dof):
        self.global_id = global_id
        self.reference_position = reference_position
        self.current_position = reference_position.copy()
        self.prescribed_dof = prescribed_dof

    def update_current_position(self):
        """Update the current position of the node."""
        # TODO implement this. For now we'll add a small perturbation to the previous deformed position
        operations.generate_random_node_current_position(node=self)


class CornerNode(Node):
    """A node located at the corner of an element.

    :param int global_id: unique global ID for the node
    :param list reference_position: reference position of the node in 3D
    :param list prescribed_dof: prescribed displacements for each degree of freedom of the node
    """

    def __init__(self, global_id, reference_position, prescribed_dof):
        super(CornerNode, self).__init__(global_id=global_id,
                                         reference_position=reference_position,
                                         prescribed_dof=prescribed_dof)


class MidpointNode(Node):
    """A node located at the midpoint of an element face.

    :param int global_id: unique global ID for the node
    :param list reference_position: reference position of the node in 3D
    :param list prescribed_dof: prescribed displacements for each degree of freedom of the node
    """

    def __init__(self, global_id, reference_position, prescribed_dof):
        super(MidpointNode, self).__init__(global_id=global_id,
                                           reference_position=reference_position,
                                           prescribed_dof=prescribed_dof)
