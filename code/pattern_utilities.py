"""Functions for working with patterns.

A pattern is defined as a 1D-array and several patterns are stored as
a 2D-array, where each row corresponds to one pattern. Each element can
take the values -1 or +1 only.
"""

import numpy as np
from scipy import stats


def generate_n_random_patterns(n_patterns, n_neurons):
    """Returns n_patterns random patterns, each of length n_neurons.

    If several patterns are generated, each row corresponds to one
    pattern.
    """
    random_0s_and_1s = stats.bernoulli.rvs(0.5, size=(n_patterns, n_neurons))
    random_minus_1s_and_1s = 2*random_0s_and_1s - 1
    return random_minus_1s_and_1s


def get_index_of_equal_pattern(pattern_to_match, patterns):
    """Returns the row index of patterns corresponding to the pattern
    that is equal to pattern_to_match.

    Returns 1 if no matching pattern is found.
    """
    for index, pattern in enumerate(patterns):
        n_different_neurons = get_n_different_neurons(pattern_to_match, pattern)
        if n_different_neurons == 0:
            return index
    return -1


def get_n_different_neurons(pattern1, pattern2):
    return sum(pattern1 != pattern2)


def vector_to_typewriter(vector, n_columns):
    """Returns 2D-array of vector.

    The first row in the returned array consists of the first n_columns
    elements in vector and so on.
    """
    return np.reshape(vector, (-1, n_columns))


def print_typewriter_pattern(pattern, n_columns):
    """Prints a pattern in a typewriter scheme."""
    print_pattern(vector_to_typewriter(pattern, n_columns))


def print_pattern(pattern):
    np.set_printoptions(formatter={"float_kind": lambda x: "%.4f" % x})
    print(repr(pattern), sep=", ")
