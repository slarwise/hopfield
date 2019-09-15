import unittest
import hopfield as pr
import pattern_utilities as utils
import numpy as np


class TestPatternRecognition(unittest.TestCase):

    def test_sign_zero_returns_one(self):
        self.assertEqual(pr.sign_zero_returns_one(-0.6), -1)
        self.assertEqual(pr.sign_zero_returns_one(0), 1)
        self.assertEqual(pr.sign_zero_returns_one(0.4), 1)

    def test_det_net_set_diagonal_weights_rule(self):
        network = pr.DeterministicHopfieldNetwork()
        network.set_diagonal_weights_rule("zero")
        self.assertTrue(network.diagonal_weights_equal_zero)
        network.set_diagonal_weights_rule("non-zero")
        self.assertFalse(network.diagonal_weights_equal_zero)
        self.assertRaises(
                KeyError,
                network.set_diagonal_weights_rule,
                *("asd",)
                )

    def test_det_net_set_patterns(self):
        patterns = utils.generate_n_random_patterns(3, 5)
        network = pr.DeterministicHopfieldNetwork()
        network.set_patterns(patterns)
        expected = (network.patterns == patterns).all()
        self.assertTrue(expected)

    def test_det_net_generate_weights_test_case_0(self):
        patterns = np.array([[1, 2], [3, 4]])
        network = pr.DeterministicHopfieldNetwork()
        network.set_diagonal_weights_rule("zero")
        network.set_patterns(patterns)
        network.generate_weights()
        expected_weights = np.array([[0, 7], [7, 0]])
        expected = (network.weights == expected_weights).all()
        self.assertTrue(expected)

    def test_det_net_generate_weights_test_case_1(self):
        patterns = np.array([[1, 2], [3, 4]])
        network = pr.DeterministicHopfieldNetwork()
        network.set_diagonal_weights_rule("non-zero")
        network.set_patterns(patterns)
        network.generate_weights()
        expected_weights = np.array([[5, 7], [7, 10]])
        expected = (network.weights == expected_weights).all()
        self.assertTrue(expected)

    def test_det_net_generate_weights_test_case_2(self):
        patterns = np.array([[-1, 1, 1], [1, -1, 1]])
        network = pr.DeterministicHopfieldNetwork()
        network.set_diagonal_weights_rule("zero")
        network.set_patterns(patterns)
        network.generate_weights()
        expected_weights = 1/3 * np.array([[0, -2, 0], [-2, 0, 0], [0, 0, 0]])
        expected = (network.weights == expected_weights).all()
        self.assertTrue(expected)

    def test_det_net_generate_weights_test_case_3(self):
        patterns = np.array([[-1, 1, 1], [1, -1, 1]])
        network = pr.DeterministicHopfieldNetwork()
        network.set_diagonal_weights_rule("non-zero")
        network.set_patterns(patterns)
        network.generate_weights()
        expected_weights = 1/3 * np.array([[2, -2, 0], [-2, 2, 0], [0, 0, 2]])
        expected = (network.weights == expected_weights).all()
        self.assertTrue(expected)

    def test_det_net_generate_weights_shape(self):
        n_patterns = 10
        n_bits = 5
        patterns = utils.generate_n_random_patterns(n_patterns, n_bits)
        network = pr.DeterministicHopfieldNetwork()
        network.set_diagonal_weights_rule("zero")
        network.set_patterns(patterns)
        network.generate_weights()
        self.assertEqual(network.weights.shape, (n_bits, n_bits))

    def test_det_net_generate_weights_diagonal(self):
        n_patterns = 6
        n_bits = 100
        patterns = utils.generate_n_random_patterns(n_patterns, n_bits)
        network = pr.DeterministicHopfieldNetwork()
        network.set_patterns(patterns)
        network.set_diagonal_weights_rule("zero")
        network.generate_weights()
        diagonal = network.weights.diagonal()
        self.assertFalse(np.any(diagonal))

    def test_det_net_update_neuron_shape(self):
        n_patterns = 5
        n_bits = 100
        patterns = utils.generate_n_random_patterns(n_patterns, n_bits)
        network = pr.DeterministicHopfieldNetwork()
        network.set_patterns(patterns)
        network.set_diagonal_weights_rule("zero")
        network.generate_weights()
        original_pattern = patterns[3, :]
        neuron_index = 10
        updated_pattern = network.update_neuron(
                original_pattern, neuron_index)
        self.assertEqual(updated_pattern.shape, original_pattern.shape)

    def test_det_net_update_neuron_difference(self):
        n_patterns = 10
        n_bits = 32
        patterns = utils.generate_n_random_patterns(n_patterns, n_bits)
        network = pr.DeterministicHopfieldNetwork()
        network.set_diagonal_weights_rule("zero")
        network.set_patterns(patterns)
        network.generate_weights()
        original_pattern = patterns[1, :]
        neuron_index = 3
        updated_pattern = network.update_neuron(
                original_pattern, neuron_index)
        n_differences = sum(updated_pattern != original_pattern)
        self.assertTrue(n_differences <= 1)

    def test_det_net_update_random_neuron(self):
        n_patterns = 20
        n_bits = 100
        patterns = utils.generate_n_random_patterns(n_patterns, n_bits)
        network = pr.DeterministicHopfieldNetwork()
        network.set_patterns(patterns)
        network.set_diagonal_weights_rule("zero")
        network.generate_weights()
        original_pattern = patterns[15, :]
        updated_pattern = network.update_random_neuron(original_pattern)
        n_differences = sum(updated_pattern != original_pattern)
        self.assertTrue(n_differences <= 1)

    def test_stoch_net_update_random_neuron(self):
        n_patterns = 20
        n_bits = 100
        patterns = utils.generate_n_random_patterns(n_patterns, n_bits)
        network = pr.StochasticHopfieldNetwork()
        network.set_patterns(patterns)
        network.set_diagonal_weights_rule("zero")
        network.set_noise_parameter(2)
        network.generate_weights()
        original_pattern = patterns[15, :]
        updated_pattern = network.update_random_neuron(original_pattern)
        n_differences = sum(updated_pattern != original_pattern)
        self.assertTrue(n_differences <= 1)

    def test_asynchronous_update_shape(self):
        n_patterns = 5
        n_bits = 30
        patterns = utils.generate_n_random_patterns(n_patterns, n_bits)
        network = pr.DeterministicHopfieldNetwork()
        network.set_patterns(patterns)
        network.set_diagonal_weights_rule("non-zero")
        network.generate_weights()
        original_pattern = patterns[0, :]
        updated_pattern = network.asynchronous_update(original_pattern, 5)
        self.assertTrue(original_pattern.shape, updated_pattern.shape)

    def test_noise_parameter(self):
        network = pr.StochasticHopfieldNetwork()
        network.set_noise_parameter(3)
        self.assertEqual(network.noise_parameter, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
