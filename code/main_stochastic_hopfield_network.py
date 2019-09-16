"""Script for estimating the order parameter."""

import numpy as np
from hopfield import StochasticHopfieldNetwork
from pattern_utilities import generate_n_random_patterns


def main():
    n_neurons = 200
    # Change n_patterns to 45 for the second question
    n_patterns = 7
    noise_parameter = 2
    n_time_steps = int(2e5)
    n_iterations = 100

    patterns = generate_n_random_patterns(n_patterns, n_neurons)

    network = StochasticHopfieldNetwork()
    network.set_patterns(patterns)
    network.set_diagonal_weights_rule("zero")
    network.set_noise_parameter(noise_parameter)
    network.generate_weights()

    pattern1 = patterns[0, :]
    updated_pattern = pattern1.copy()

    m = np.zeros(n_iterations)
    for i in range(n_iterations):
        for _ in range(n_time_steps):
            updated_pattern = network.update_random_neuron(updated_pattern)
            m[i] += np.inner(updated_pattern, pattern1)
        m[i] /= n_neurons
        m[i] /= n_time_steps

    m_estimate = sum(m) / n_iterations
    print("{:.3f}".format(m_estimate))


if __name__ == "__main__":
    main()
