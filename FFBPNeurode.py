"""
Feedforward backpropagation neurode class. To be used with the neural
network, which is implemented using individual neurodes.
"""

from __future__ import annotations
from enum import Enum
import numpy as np
import random
from abc import ABC, abstractmethod


class MultiLinkNode(ABC):
    """Abstract base class as starting point for FFBPNeurode class.

    Attributes:
        _reporting_nodes (dict): utilizes binary encoding to keep track
            of which neighboring nodes have indicated that they have
            information available. Keys are elements of Side enum,
            values are the binary encodings.
        _reference_value (dict): indicates what the _reporting_nodes
            value should be when all nodes have reported. Keys are
            elements of Side enum, values are the binary encodings.
        _neighbors (dict): contains references to neighboring nodes.
            Keys are elements of Side enum, values are the references.
    """
    def __init__(self):
        """Initialize the class instance."""
        self._reporting_nodes = {MultiLinkNode.Side.UPSTREAM: 0,
                                 MultiLinkNode.Side.DOWNSTREAM: 0}
        self._reference_value = {MultiLinkNode.Side.UPSTREAM: 0,
                                 MultiLinkNode.Side.DOWNSTREAM: 0}
        self._neighbors = {MultiLinkNode.Side.UPSTREAM: [],
                           MultiLinkNode.Side.DOWNSTREAM: []}

    def __str__(self) -> str:
        """Print ID of node and all neighboring nodes.

        Returns:
            str: string representation of node and neighboring nodes
        """
        us_nodes = self._neighbors[MultiLinkNode.Side.UPSTREAM]
        ds_nodes = self._neighbors[MultiLinkNode.Side.DOWNSTREAM]
        rows = max(len(us_nodes), len(ds_nodes))
        string = f"Upstream Nodes{' ' * 6}" \
                 f"Node{' ' * 16}" \
                 f"Downstream Nodes{' ' * 4}\n"
        for i in range(rows):
            string += f"{id(us_nodes[i]) if i < len(us_nodes) else '':<20}" \
                      f"{id(self) if i == 0 else '':<20}" \
                      f"{id(ds_nodes[i]) if i < len(ds_nodes) else '':<20}\n"
        return string

    class Side(Enum):
        UPSTREAM = 0
        DOWNSTREAM = 1

    def reset_neighbors(self, nodes: list, side: Side):
        """Reset nodes that link into this node.

        Args:
            nodes (list): nodes to set as neighbors
            side (Side): upstream or downstream
        """
        self._neighbors[side] = nodes.copy()
        for node in nodes:
            self._process_new_neighbor(node, side)
        self._reference_value[side] = 2**len(nodes) - 1

    @abstractmethod
    def _process_new_neighbor(self, node: MultiLinkNode, side: Side):
        pass


class Neurode(MultiLinkNode):
    """Neurode class that inherits from MultilinkNode.

    Attributes:
        _value (float): current value of neurode
        _node_type (LayerType): input, hidden, or output node
        _learning_rate (float): learning rate for backpropogation
        _weights (dict): represents weights for upstream connections.
            Keys are references to upstream neurodes, values are floats
            representing weights
    """

    def __init__(self, node_type: LayerType, learning_rate=0.05):
        """Initialize the class instance.

        Args:
            node_type (LayerType): input, hidden, or output node
            learning_rate (float): learning rate for backpropogation
        """
        super().__init__()
        self._value = 0
        self._node_type = node_type
        self._learning_rate = learning_rate
        self._weights = {}

    @property
    def value(self) -> float:
        return self._value

    @property
    def node_type(self) -> LayerType:
        return self._node_type

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate: float):
        self._learning_rate = learning_rate

    def get_weight(self, node: Neurode) -> float:
        return self._weights[node]

    def _process_new_neighbor(self, node: Neurode,
                              side: MultiLinkNode.Side):
        """Add upstream node references to _weights dictionary.

        This method is called when new neighbors are added.

        Args:
            node (Neurode): node to process
            side (MultiLinkNode.Side): upstream or downstream
        """
        if side is MultiLinkNode.Side.UPSTREAM:
            self._weights[node] = random.random()

    def _check_in(self, node: Neurode, side: MultiLinkNode.Side) -> bool:
        """Update and evaluate values in _reporting_nodes.

        This method is called whenever the node learns that a
        neighboring node has information available. It determines the
        index of that node and updates _reporting_nodes to reflect that
        the node has been reported. The new value is then compared to
         _reference_value to determine whether all neighboring nodes
        have reported.

        Args:
            node (Neurode): neighboring node with available
                information
            side (MultiLinkNode.Side): upstream or downstream
        Returns:
            bool: True if updated _reporting_nodes value matches
                _reference_value else False
        """
        node_idx = self._neighbors[side].index(node)
        self._reporting_nodes[side] = self._reporting_nodes[side] | 2**node_idx
        if self._reporting_nodes[side] == self._reference_value[side]:
            self._reporting_nodes[side] = 0
            return True
        else:
            return False


class FFNeurode(Neurode):
    """Feedforward Neurode.

    This class handles the feedforward implementation for our neurode.
    """

    def __init__(self, my_type: LayerType):
        """Initialize the class instance.

        Args:
            my_type (LayerType): input, hidden, or output node
        """
        super().__init__(my_type)

    def set_input(self, input_value: float):
        """Set value for input layer neurodes and fire downstream.

        Args:
            input_value (float): value to input
        """
        self._value = input_value
        self._fire_downstream()

    def data_ready_upstream(self, node: Neurode):
        """Check upstream neurode information and fire if ready.

        This method is called by an upstream neighbor onto downstream
        neighbors to let them know that upstream data is ready. If all
        of the current neurode's upstream neighbors have data available,
        then it too processes its own data and fires downstream.

        Args:
            node (Neurode): upstream neurode that called this function
                for the current neurode
        """
        if self._check_in(node, MultiLinkNode.Side.UPSTREAM):
            self._calculate_value()
            self._fire_downstream()

    def _fire_downstream(self):
        """Indicate downstream that information is available.

        This method is called whenever the node is ready to relay
        information to downstream neighbors. It calls
        data_ready_upstream for each downstream neighbor, passing
        self as an argument.
        """
        for ds_neighbor in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            ds_neighbor.data_ready_upstream(self)

    def _calculate_value(self):
        """Calculate and update neurode value.

        Calculate the result of the sigmoid function with an
        input of the weighted sum of values from upstream neighbors
        and store result in _value.
        """
        us_weighted_sum = sum([us_neurode.value * weight for us_neurode, weight
                               in self._weights.items()])
        self._value = self._sigmoid(us_weighted_sum)

    @staticmethod
    def _sigmoid(value: float) -> float:
        """Calculate the result of the sigmoid function at a value.

        Args:
            value (float): value to input into the sigmoid function
        Returns:
            float: result of sigmoid function at given value
        """
        return 1 / (1 + np.exp(-value))


class BPNeurode(Neurode):
    """Backpropagation Neurode.

    This class handles the backpropagation implementation for our
    neurode.

    Attributes:
        _delta (float): current delta value of neurode
    """

    def __init__(self, my_type: LayerType):
        """Initialize the class instance.

        Args:
            my_type (LayerType): input, hidden, or output node
        """
        super().__init__(my_type)
        self._delta = 0

    @property
    def delta(self) -> float:
        return self._delta

    def data_ready_downstream(self, node: Neurode):
        """Check downstream neurode information and fire if ready.

        This method is called by a downstream neighbor onto upstream
        neighbors to let them know that downstream data is ready. If
        all of the current neurode's downstream neighbors have data
        available, then it too processes its own data, fires upstream,
        and updates downstream weights.

        Args:
            node (Neurode): downstream neurode that called this function
                for the current neurode
        """
        if self._check_in(node, MultiLinkNode.Side.DOWNSTREAM):
            self._calculate_delta()
            self._fire_upstream()
            self._update_weights()

    def set_expected(self, expected_value: float):
        """Set value for output layer neurode and fire upstream.

        Expected value will be used to calculate and store the node's
        delta value.

        Args:
            expected_value (float): expected value
        """
        self._calculate_delta(expected_value)
        self._fire_upstream()

    def adjust_weights(self, node: Neurode, adjustment: float):
        """Adjust the weight between self and an upstream neighbor.

        This method is called by an upstream neighbor onto a downstream
        neighbor to adjust the weight associated between them.

        Args:
            node (Neurode): upstream neurode that called this function
                for the current neurode
            adjustment (float): amount to adjust the weight by
        """
        self._weights[node] += adjustment

    def _update_weights(self):
        """Update weights between self and downstream neighbors.

        The adjustment to each weight is the result of the upstream
        neurode's value multiplied by the downstream neurode's delta
        multiplied by the downstream neurode's learning rate.
        """
        for ds_neighbor in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]:
            adjustment = self.value * ds_neighbor.delta \
                         * ds_neighbor.learning_rate
            ds_neighbor.adjust_weights(self, adjustment)

    def _fire_upstream(self):
        """Indicate upstream that information is available.

        This method is called whenever the node is ready to relay
        information to upstream neighbors. It calls
        data_ready_downstream for each upstream neighbor, passing
        self as an argument.
        """
        for us_neighbor in self._neighbors[MultiLinkNode.Side.UPSTREAM]:
            us_neighbor.data_ready_downstream(self)

    def _calculate_delta(self, expected_value: float = None):
        """Calculate and update neurode delta value.

        Case 1: Hidden layer neurode: no expected value argument given:
            Calculate the delta value as the weighted sum of downstream
            delta values multiplied by the node's sigmoid derivative
            value. Store result in _delta.

        Case 2: Output layer neurode: expected value argument given:
            Calculate the delta value as the error between the expected
            value and the node's value, multiplied by the node's sigmoid
            derivative value. Store result in _delta.
        """
        if expected_value is None:
            ds_weighted_sum = sum(
                [ds_neurode.get_weight(self) * ds_neurode.delta
                 for ds_neurode
                 in self._neighbors[MultiLinkNode.Side.DOWNSTREAM]])
            self._delta = ds_weighted_sum * self._sigmoid_derivative(self.value)
        else:
            self._delta = (expected_value - self.value) * \
                          self._sigmoid_derivative(self.value)

    @staticmethod
    def _sigmoid_derivative(value: float) -> float:
        """Calculate the sigmoid derivative.

        This function will calculate the sigmoid derivative given an
        input value that has been passed through the sigmoid function.

        Args:
            value (float): value that has been passed through the
            sigmoid function
        Returns:
            float: sigmoid derivative
        """
        return value * (1 - value)


class FFBPNeurode(FFNeurode, BPNeurode):
    """Feedforward Backpropagation Neurode."""
    pass
