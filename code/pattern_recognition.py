import numpy as np
from scipy import stats
from abc import ABC, abstractmethod


class HopfieldNetwork(ABC):
    def set_diagonal_weights_rule(self, diagonal_weights_rule):
        if diagonal_weights_rule == "zero":
            self.diagonal_weights_equal_zero = True
        elif diagonal_weights_rule == "non-zero":
            self.diagonal_weights_equal_zero = False
        else:
            raise KeyError(
                    diagonal_weights_rule
                    + " not a valid diagonal_weights_rule"
                    )

    def set_patterns(self, patterns):
        self.patterns = patterns

    def generate_weights(self):
        _, n_bits = self.patterns.shape
        self.weights = np.zeros((n_bits, n_bits))
        for pattern in self.patterns:
            self.weights += np.outer(pattern, pattern)
        self.weights /= n_bits

        if self.diagonal_weights_equal_zero:
            np.fill_diagonal(self.weights, 0)

    def asynchronous_update(self, pattern, n_updates):
        updated_pattern = pattern.copy()
        for i in range(n_updates):
            updated_pattern = self.update_random_neuron(updated_pattern)
        return updated_pattern

    def update_random_neuron(self, pattern):
        n_bits = pattern.shape
        neuron_index = np.random.randint(n_bits)
        return self.update_neuron(pattern, neuron_index)

    def update_neuron(self, pattern, neuron_index):
        weights_i = self.weights[neuron_index, :]
        local_field = np.inner(weights_i, pattern)
        updated_bit = self.get_state_of_local_field(local_field)
        updated_pattern = pattern.copy()
        updated_pattern[neuron_index] = updated_bit
        return updated_pattern

    @abstractmethod
    def get_state_of_local_field(self, local_field):
        pass


class DeterministicHopfieldNetwork(HopfieldNetwork):
    def get_state_of_local_field(self, local_field):
        return sign_zero_returns_one(local_field)


class StochasticHopfieldNetwork(HopfieldNetwork):
    def get_state_of_local_field(self, local_field):
        p = 1/(1 + np.exp(-2*self.noise_parameter*local_field))
        rand = stats.bernoulli.rvs(p)
        return 1 if rand else -1

    def set_noise_parameter(self, noise_parameter):
        self.noise_parameter = noise_parameter


def sign_zero_returns_one(value):
    return 1 if value >= 0 else -1
