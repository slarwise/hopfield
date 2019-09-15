from pattern_utilities import generate_n_random_patterns
from hopfield import StochasticHopfieldNetwork
import numpy as np


def main():
    n_bits = 200
    n_patterns = 7
    noise_parameter = 2
    T = int(2e5)
    n_iterations = 100

    patterns = generate_n_random_patterns(n_patterns=n_patterns, n_bits=n_bits)

    network = StochasticHopfieldNetwork()
    network.set_patterns(patterns)
    network.set_diagonal_weights_rule("zero")
    network.set_noise_parameter(noise_parameter)
    network.generate_weights()

    pattern1 = patterns[0, :]
    updated_pattern = pattern1.copy()

    m = np.zeros(n_iterations)
    for i in range(n_iterations):
        print(i)
        for t in range(T):
            updated_pattern = network.update_random_neuron(updated_pattern)
            m[i] += np.inner(updated_pattern, pattern1)
        m[i] /= n_bits
        m[i] /= T

    m_estimate = sum(m) / n_iterations
    print("{:.3f}".format(m_estimate))


if __name__ == "__main__":
    main()
