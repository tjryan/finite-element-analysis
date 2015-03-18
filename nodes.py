"""
nodes.py contains the nodes for the finite elements.

.. moduleauthor:: Tyler Ryan <tyler.ryan@engineering.ucla.edu>
"""
import numpy
import operations


class Node:
    """Point in space that defines the boundaries of an element and the connection points between elements.
    Also defines where the degrees of freedom are defined, such as displacements and rotations due to loading.
    """

    def __init__(self):
        self.local_id = None  # local identifier for the node
        self.global_id = None  # global identifier for the node
        self.reference_position = []
        self.current_position = []
        self.parent_elements = []  # the elements object that this node belongs to

    def update_current_position(self):
        """Update the current position of the node."""
        # TODO implement this. For now we'll add a small perturbation to the previous deformed position
        operations.generate_random_node_current_position(node=self)


class CornerNode(Node):
    """A node located at the corner of an element."""

    def __init__(self):
        super(CornerNode, self).__init__()


class MidpointNode(Node):
    """A node located at the midpoint of an element face."""

    def __init__(self):
        super(MidpointNode, self).__init__()
