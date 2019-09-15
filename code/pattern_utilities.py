import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def generate_n_random_patterns(n_patterns, n_bits):
    random_0s_and_1s = stats.bernoulli.rvs(0.5, size=(n_patterns, n_bits))
    random_minus_1s_and_1s = 2*random_0s_and_1s - 1
    return random_minus_1s_and_1s


def get_index_of_equal_pattern(pattern_to_match, patterns):
    for index, pattern in enumerate(patterns):
        n_different_bits = get_n_different_bits(pattern_to_match, pattern)
        if n_different_bits == 0:
            return index
    return -1


def get_n_different_bits(pattern1, pattern2):
    return sum(pattern1 != pattern2)


def vector_to_typewriter(vector, n_columns):
    return np.reshape(vector, (-1, n_columns))


def print_typewriter_pattern(pattern, n_columns):
    print_pattern(vector_to_typewriter(pattern, n_columns))


def print_pattern(pattern):
    np.set_printoptions(formatter={"float_kind": lambda x: "%.4f" % x})
    print(repr(pattern), sep=", ")


def plot_pattern(pattern):
    plt.imshow(pattern, cmap="Greys")
    plt.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            left=False,
            labelbottom=False,
            labelleft=False)
