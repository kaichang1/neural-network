"""
Neural network class. The structure of our neural network is managed
with a doubly linked list, with each list node representing different
layers of the network.
"""

from __future__ import annotations
from typing import TypeVar

from NNData import NNData
from FFBPNeurode import MultiLinkNode, FFBPNeurode, LayerType

T = TypeVar('T')


class Node:
    """Linked List Node.

    Attributes:
        data (T): data associated with node
        next (Node): next neighbor node
        prev (Node): previous neighbor node
    """

    def __init__(self, data: T = None):
        """Initialize the class instance.

        Args:
            data (T): data to associate with node
        """
        self.data = data
        self.next = None
        self.prev = None


class DoublyLinkedList:
    """Doubly Linked List.

    Attributes:
        _head (Node): head node
        _tail (Node): tail node
        _cur (Node): current node
    """

    def __init__(self):
        """Initialize the class instance."""
        self._head = None
        self._tail = None
        self._cur = None

    def __iter__(self) -> DoublyLinkedList:
        self._cur_iter = self._head
        return self

    def __next__(self) -> T:
        if self._cur_iter is None:
            raise StopIteration
        ret_val = self._cur_iter.data
        self._cur_iter = self._cur_iter.next
        return ret_val

    class EmptyListError(Exception):
        pass

    def get_current_data(self) -> T:
        if self._head is None:
            raise DoublyLinkedList.EmptyListError
        return self._cur.data

    def add_to_head(self, data: T):
        """Create and add node to head of linked list.

        Args:
            data (T): data to associate with node
        """
        new_node = Node(data)
        new_node.next = self._head
        if self._head:
            self._head.prev = new_node
        else:
            self._tail = new_node
        self._head = new_node
        self.reset_to_head()

    def remove_from_head(self) -> T:
        """Remove node from head of linked list.

        Returns:
            T: data value of old head node
        """
        if self._head is None:
            raise DoublyLinkedList.EmptyListError
        ret_val = self._head.data
        self._head = self._head.next
        if self._head:
            self._head.prev = None
            self.reset_to_head()
        else:
            self._tail = None
            self._cur = None
        return ret_val

    def add_after_cur(self, data: T):
        """Create and add node after current node.

        Args:
            data: data value to associate with node
        """
        if self._cur is None:
            self.add_to_head(data)
            return
        new_node = Node(data)
        new_node.next = self._cur.next
        if new_node.next:
            new_node.next.prev = new_node
        else:
            self._tail = new_node
        self._cur.next = new_node
        new_node.prev = self._cur

    def remove_after_cur(self) -> T:
        """Remove node after current node.

        Returns:
            T: data value of removed node
        """
        if self._head is None:
            raise DoublyLinkedList.EmptyListError
        if self._cur.next is None:
            raise IndexError
        ret_val = self._cur.next.data
        self._cur.next = self._cur.next.next
        if self._cur.next:
            self._cur.next.prev = self._cur
        else:
            self._tail = self._cur
        return ret_val

    def reset_to_head(self) -> T:
        """Reset current node to head node.

        Returns:
            T: data value of head node
        """
        self._cur = self._head
        if self._cur is None:
            raise DoublyLinkedList.EmptyListError
        else:
            return self._cur.data

    def reset_to_tail(self) -> T:
        """Reset current node to tail node.

        Returns:
            T: data value of tail node
        """
        self._cur = self._tail
        if self._cur is None:
            raise DoublyLinkedList.EmptyListError
        else:
            return self._cur.data

    def move_forward(self) -> T:
        """Move current node one node forward.

        Returns:
            T: data value of new current node
        """
        if self._head is None:
            raise DoublyLinkedList.EmptyListError
        if self._cur.next is None:
            raise IndexError
        self._cur = self._cur.next
        return self._cur.data

    def move_back(self) -> T:
        """Move current node one node back.

        Returns:
            T: data value of new current node
        """
        if self._head is None:
            raise DoublyLinkedList.EmptyListError
        if self._cur.prev is None:
            raise IndexError
        self._cur = self._cur.prev
        return self._cur.data


class LayerList(DoublyLinkedList):
    """Neural network layer list.

    This class handles the layout of our neural network via a doubly
    linked list. Each node represents a neural network layer. Node
    values are lists of references to FFBPNeurodes within that layer.
    The head node represents the input layer, while the tail node
    represents the output layer.
    """

    def __init__(self, inputs: int, outputs: int):
        """Initialize the class instance.

        Create input and output layers with their associated neurodes.
        Link these layers together.

        Args:
            inputs (int): number of input layer neurodes
            outputs (int): number of output layer neurodes
        """
        super().__init__()
        input_neurodes = [FFBPNeurode(LayerType.INPUT)
                          for _ in range(inputs)]
        output_neurodes = [FFBPNeurode(LayerType.OUTPUT)
                           for _ in range(outputs)]

        for input_neurode in input_neurodes:
            input_neurode.reset_neighbors(output_neurodes,
                                          MultiLinkNode.Side.DOWNSTREAM)
        for output_neurode in output_neurodes:
            output_neurode.reset_neighbors(input_neurodes,
                                           MultiLinkNode.Side.UPSTREAM)

        self.add_to_head(input_neurodes)
        self.add_after_cur(output_neurodes)

    @property
    def input_nodes(self) -> list[FFBPNeurode]:
        return self._head.data

    @property
    def output_nodes(self) -> list[FFBPNeurode]:
        return self._tail.data

    def add_layer(self, num_nodes: int):
        """Add hidden layer to neural network.

        Args:
            num_nodes (int): number of hidden layer neurodes
        """
        if self._cur == self._tail:
            raise IndexError
        cur_neurodes = self._cur.data
        hidden_neurodes = [FFBPNeurode(LayerType.HIDDEN)
                           for _ in range(num_nodes)]
        next_neurodes = self._cur.next.data

        for cur_neurode in cur_neurodes:
            cur_neurode.reset_neighbors(hidden_neurodes,
                                        MultiLinkNode.Side.DOWNSTREAM)
        for hidden_neurode in hidden_neurodes:
            hidden_neurode.reset_neighbors(cur_neurodes,
                                           MultiLinkNode.Side.UPSTREAM)
            hidden_neurode.reset_neighbors(next_neurodes,
                                           MultiLinkNode.Side.DOWNSTREAM)
        for next_neurode in next_neurodes:
            next_neurode.reset_neighbors(hidden_neurodes,
                                         MultiLinkNode.Side.UPSTREAM)

        self.add_after_cur(hidden_neurodes)

    def remove_layer(self):
        """Remove neural network hidden layer after current layer node."""
        if self._cur.next == self._tail:
            raise IndexError
        self.remove_after_cur()
        cur_neurodes = self._cur.data
        next_neurodes = self._cur.next.data

        for cur_neurode in cur_neurodes:
            cur_neurode.reset_neighbors(next_neurodes,
                                        MultiLinkNode.Side.DOWNSTREAM)
        for next_neurode in next_neurodes:
            next_neurode.reset_neighbors(cur_neurodes,
                                         MultiLinkNode.Side.UPSTREAM)


class FFBPNetwork:
    """Feedforward Backpropagation Neural Network.

    Attributes:
        _network (LayerList): neural network
        _num_inputs (int): number of inputs for neural network
        _num_outputs (int): number of outputs for neural network
    """

    def __init__(self, num_inputs: int, num_outputs: int):
        """Initialize the class instance.

        Args:
            num_inputs (int): number of inputs for neural network
            num_outputs (int): number of outputs for neural network
        """
        self._network = LayerList(num_inputs, num_outputs)
        self._num_inputs = num_inputs
        self._num_outputs = num_outputs

    class EmptySetException(Exception):
        pass

    def add_hidden_layer(self, num_nodes: int, position: int = 0):
        """Add hidden layer to neural network.

        Args:
            num_nodes (int): number of nodes in hidden layer
            position (int): position within neural network to add hidden
                layer. A position of 0 adds a hidden layer immediately
                after the input layer. Each integer increment above 0
                moves the hidden layer one layer forward.
        """
        self._network.reset_to_head()
        for i in range(position):
            self._network.move_forward()
        self._network.add_layer(num_nodes)

    def train(self, data_set: NNData, epochs: int = 1000, verbosity: int = 2,
              order=NNData.Order.RANDOM):
        """Train the neural network.

        Root mean square error (RMSE) calculated for every epoch and
        reported based on verbosity value.

        Args:
            data_set (NNData): neural network dataset
            epochs (int): number of epochs to run
            verbosity (int): specifies reporting frequency:
                0: no reports made
                1: RMSE of most recent epoch reported every 100 epochs
                2: RMSE of most recent epoch reported every 100 epochs
                    and print out all input, expected, and output values
                    for every training example every 1000 epochs
            order (NNData.Order): random or sequential ordering
        """
        if data_set.number_of_samples(NNData.Set.TRAIN) == 0:
            raise FFBPNetwork.EmptySetException

        rmse = None
        for epoch_num in range(epochs):
            sq_error = 0
            data_set.prime_data(NNData.Set.TRAIN, order)
            while not data_set.pool_is_empty(NNData.Set.TRAIN):
                feature_label = data_set.get_one_item(NNData.Set.TRAIN)

                # Use features as inputs to input layer neurodes and
                # begin feedforward process
                for i in range(len(feature_label[0])):
                    input_neurode = self._network.input_nodes[i]
                    input_neurode.set_input(feature_label[0][i])
                # Use labels as expected values for output layer
                # neurodes and begin backpropagation process
                for i in range(len(feature_label[1])):
                    output_neurode = self._network.output_nodes[i]
                    output_neurode.set_expected(feature_label[1][i])
                    error = feature_label[1][i] - output_neurode.value
                    sq_error += error**2

                if (verbosity > 1) and (epoch_num % 1000 == 0):
                    inputs = feature_label[0]
                    expected = feature_label[1]
                    outputs = [output_neurode.value for output_neurode in
                               self._network.output_nodes]
                    print(f"Input{inputs} Expected{expected} Output{outputs}")

            total_output_nodes = data_set.number_of_samples(NNData.Set.TRAIN) * \
                               len(self._network.output_nodes)
            rmse = (sq_error / total_output_nodes)**(1/2)
            if (verbosity > 0) and (epoch_num % 100 == 0):
                print(f"Epoch {epoch_num} RMSE: {rmse}")

        print(f"Final RMSE: {rmse}")

    def test(self, data_set: NNData, order=NNData.Order.SEQUENTIAL):
        """Test the neural network.

        Root mean square error (RMSE) calculated at the end of the test.

        Args:
            data_set (NNData): neural network dataset
            order (NNData.Order): random or sequential ordering
        """
        if data_set.number_of_samples(NNData.Set.TEST) == 0:
            raise FFBPNetwork.EmptySetException

        sq_error = 0
        data_set.prime_data(NNData.Set.TEST, order)
        while not data_set.pool_is_empty(NNData.Set.TEST):
            feature_label = data_set.get_one_item(NNData.Set.TEST)

            # Use features as inputs to input layer neurodes and
            # begin feedforward process
            for i in range(len(feature_label[0])):
                input_neurode = self._network.input_nodes[i]
                input_neurode.set_input(feature_label[0][i])
            # No backpropagation in testing
            for i in range(len(feature_label[1])):
                output_neurode = self._network.output_nodes[i]
                error = feature_label[1][i] - output_neurode.value
                sq_error += error**2

            inputs = feature_label[0]
            expected = feature_label[1]
            outputs = [output_neurode.value for output_neurode in
                       self._network.output_nodes]
            print(f"Input{inputs} Expected{expected} Output{outputs}")

        total_output_nodes = data_set.number_of_samples(NNData.Set.TEST) * \
                             len(self._network.output_nodes)
        rmse = (sq_error / total_output_nodes)**(1/2)
        print(f"Final RMSE: {rmse}")
