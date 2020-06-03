import abc
import numpy as np


class Agent(abc.ABC):

    @abc.abstractmethod
    def train(self, episodes):
        pass

    @abc.abstractmethod
    def evaluate(self, sentence):
        pass

    @abc.abstractmethod
    def choose_action(self, actions, softmax=False, axis=None):
        """
        Get the softmax or argmax action
        .
        :param actions: ND-array of action values.
        :param softmax: flag to select type of sampling.
        :param axis: aggregate along which axis.
        :return: integer action
        """
        if softmax:
            # subtract by max value for numerical stability.
            e_x = np.exp(actions - np.max(actions))
            p_a = e_x / e_x.sum(axis=axis, keepdims=True)
            return np.random.choice(len(p_a), 1, p=p_a)[0]
        else:
            indexes = np.argmax(actions, axis=axis)
            return np.random.choice(indexes, 1)[0]
