import unittest
import pattern_utilities as utils
import numpy as np


class TestPatternUtilities(unittest.TestCase):

    def test_generate_n_random_patterns_shape(self):
        n_patterns = 2
        n_bits = 3
        patterns = utils.generate_n_random_patterns(n_patterns, n_bits)
        self.assertEqual(patterns.shape, (n_patterns, n_bits))

    def test_generate_n_random_patterns_values(self):
        n_patterns = 10
        n_bits = 5
        patterns = utils.generate_n_random_patterns(n_patterns, n_bits)
        unique_values = np.unique(patterns)
        expected = (unique_values == np.array([-1, 1])).all()
        self.assertTrue(expected)


if __name__ == "__main__":
    unittest.main(verbosity=0)
