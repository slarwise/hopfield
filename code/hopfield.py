"""Implementation of a Hopfield network using Hebb's rule."""

from abc import ABC, abstractmethod
import numpy as np
from scipy import stats


class HopfieldNetwork(ABC):
    """Abstract Hopfield net using Hebb's rule to compute weights."""

    def set_diagonal_weights_rule(self, diagonal_weights_rule):
        """Specify if the diagonal weights should be zero or not.

        diagonal_weights_rule must equal "zero" or "non-zero".
        """
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
        """Specify the stored patterns.

        patterns must be a 1D- or 2D-array where each row is a pattern.
        """
        self.patterns = patterns

    def generate_weights(self):
        """Generate the weights for the network."""
        _, n_neurons = self.patterns.shape
        self.weights = np.zeros((n_neurons, n_neurons))
        for pattern in self.patterns:
            self.weights += np.outer(pattern, pattern)
        self.weights /= n_neurons

        if self.diagonal_weights_equal_zero:
            np.fill_diagonal(self.weights, 0)

    def update_random_neuron(self, pattern):
        n_neurons = pattern.shape
        neuron_index = np.random.randint(n_neurons)
        return self.update_neuron(pattern, neuron_index)

    def update_neuron(self, pattern, neuron_index):
        """Returns updated pattern after update of neuron_index."""
        weights_i = self.weights[neuron_index, :]
        local_field = np.inner(weights_i, pattern)
        updated_neuron = self.get_state_of_local_field(local_field)
        updated_pattern = pattern.copy()
        updated_pattern[neuron_index] = updated_neuron
        return updated_pattern

    @abstractmethod
    def get_state_of_local_field(self, local_field):
        pass


class DeterministicHopfieldNetwork(HopfieldNetwork):
    """Concrete Hopfield net with deterministic updating."""

    def get_state_of_local_field(self, local_field):
        """"Update rule when updating a neuron."""
        return 1 if local_field >= 0 else -1


class StochasticHopfieldNetwork(HopfieldNetwork):
    """Concrete Hopfield net with stochastic updating."""

    def get_state_of_local_field(self, local_field):
        """"Update rule when updating a neuron.

        Returns 1 with probability
        1 / (1+exp(-2*noise_parameter*local_field),
        else -1.
        """
        p = 1/(1 + np.exp(-2*self.noise_parameter*local_field))
        rand = stats.bernoulli.rvs(p)
        return 1 if rand else -1

    def set_noise_parameter(self, noise_parameter):
        self.noise_parameter = noise_parameter
