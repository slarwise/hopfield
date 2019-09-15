import numpy as np
from pattern_recognition import DeterministicHopfieldNetwork
from pattern_utilities import generate_n_random_patterns, print_pattern


def main():
    n_bits = 120
    n_patterns_vector = [12, 24, 48, 70, 100, 120]
    diagonal_weights_rule = "non-zero"
    n_iterations = int(1e5)
    one_step_error_probability = np.zeros(len(n_patterns_vector))

    for n_patterns_i, n_patterns in enumerate(n_patterns_vector):
        n_errors = 0
        for i in range(n_iterations):
            network = DeterministicHopfieldNetwork()

            patterns = generate_n_random_patterns(n_patterns, n_bits)
            network.set_patterns(patterns)
            network.set_diagonal_weights_rule(diagonal_weights_rule)
            network.generate_weights()

            pattern_to_feed_index = np.random.randint(n_patterns)
            original_pattern = patterns[pattern_to_feed_index, :]
            neuron_to_update_index = np.random.randint(n_bits)

            updated_pattern = network.update_neuron(
                    original_pattern,
                    neuron_to_update_index
                    )
            updated_neuron = updated_pattern[neuron_to_update_index]
            original_neuron = original_pattern[neuron_to_update_index]
            updated_neuron = updated_pattern[neuron_to_update_index]

            if original_neuron != updated_neuron:
                n_errors += 1

        one_step_error_probability[n_patterns_i] = n_errors/n_iterations

    print_pattern(one_step_error_probability)


if __name__ == "__main__":
    main()
