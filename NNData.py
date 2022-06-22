"""
Neural network dataset class. To be used with the neural network.
Training and test sets are represented using indirect indices that are
randomly generated.
"""

from collections import deque
from typing import Optional
from enum import Enum
import numpy as np
import random


class NNData:
    """Neural network data class to manage training and testing data.

    Attributes:
        _features (np.array): represents features set
        _labels (np.array): represents labels set
        _train_factor (float): percentage of data used for training set
        _train_indices (list): points to items in training set
        _test_indices (list): points to items in testing set
        _train_pool (deque): keeps track of training items not yet seen
            in training run
        _test_pool (deque): keeps track of testing items not yet seen
            in testing run
    """

    def __init__(self, features: list[list] = None, labels: list[list] = None,
                 train_factor=0.9):
        """Initialize the class instance.

        Args:
            features (list[list]): features where each row represents
                one example
            labels (list[list]): labels where each row represents one
                example
            train_factor (float): percentage of data used for training
                set
        """
        if features is None:
            features = []
        if labels is None:
            labels = []
        self._features = None
        self._labels = None
        self._train_factor = NNData.percentage_limiter(train_factor)
        self._train_indices = []
        self._test_indices = []
        self._train_pool = deque()
        self._test_pool = deque()

        try:
            self.load_data(features, labels)
        except (DataMismatchError, ValueError):
            pass
        self.split_set()

    class Order(Enum):
        RANDOM = 0
        SEQUENTIAL = 1

    class Set(Enum):
        TRAIN = 0
        TEST = 1

    def load_data(self, features: list[list] = None,
                  labels: list[list] = None):
        """Load features and labels.

        Args:
            features (list[list]): features where each row represents
                one example
            labels (list[list]): labels where each row represents one
                example
        """
        if features is None:
            self._features = None
            self._labels = None
            return
        if len(features) != len(labels):
            self._features = None
            self._labels = None
            raise DataMismatchError
        try:
            self._features = np.array(features, dtype=float)
            self._labels = np.array(labels, dtype=float)
        except ValueError:
            self._features = None
            self._labels = None
            raise ValueError

    def prime_data(self, target_set: Set = None, order: Order = None):
        """Load deques to be used as indirect indices.

        Args:
            target_set (Set): training or test set
            order (Order): random or sequential ordering
        """
        if target_set is NNData.Set.TRAIN or target_set is None:
            self._train_pool = deque(self._train_indices)
            if order is NNData.Order.RANDOM:
                random.shuffle(self._train_pool)
        if target_set is NNData.Set.TEST or target_set is None:
            self._test_pool = deque(self._test_indices)
            if order is NNData.Order.RANDOM:
                random.shuffle(self._test_pool)

    def get_one_item(self, target_set: Set = None) -> Optional[tuple]:
        """Return one feature/label pair.

        Args:
            target_set (Set): training or test set
        Returns:
            tuple: feature/label pair
        """
        if self.pool_is_empty(target_set):
            return None
        elif target_set is NNData.Set.TRAIN or target_set is None:
            idx = self._train_pool.popleft()
            return self._features[idx], self._labels[idx]
        elif target_set is NNData.Set.TEST:
            idx = self._test_pool.popleft()
            return self._features[idx], self._labels[idx]

    def number_of_samples(self, target_set: Set = None) -> int:
        """Return total number of examples in target set.

        Args:
            target_set (Set): training or test set
        Returns:
            int: total number of examples
        """
        if target_set is NNData.Set.TRAIN:
            return len(self._train_indices)
        elif target_set is NNData.Set.TEST:
            return len(self._test_indices)
        elif target_set is None:
            return len(self._train_indices) + len(self._test_indices)

    def pool_is_empty(self, target_set: Set = None) -> bool:
        """Return True if target set deque is empty.

        Args:
            target_set (Set): training or test set
        Returns:
            bool: True if deque is empty
        """
        if target_set is NNData.Set.TRAIN or target_set is None:
            return not bool(self._train_pool)
        elif target_set is NNData.Set.TEST:
            return not bool(self._test_pool)

    def split_set(self, new_train_factor: float = None):
        """Create indirect indices for training and test sets.

        Args:
            new_train_factor (float): new train factor
        """
        if new_train_factor is not None:
            self._train_factor = NNData.percentage_limiter(new_train_factor)
        num_examples = len(self._labels)
        training_size = int(num_examples * self._train_factor)

        indices = [i for i in range(num_examples)]
        random.shuffle(indices)
        self._train_indices = sorted(indices[:training_size])
        self._test_indices = sorted(indices[training_size:])

    @staticmethod
    def percentage_limiter(percentage: float) -> float:
        """Limit input values to between 0 and 1.

        Args:
            percentage (float): value to limit
        Returns:
            float: value limited between 0 and 1
        """
        if percentage < 0:
            return 0
        elif percentage > 1:
            return 1
        else:
            return percentage


class DataMismatchError(Exception):
    pass
