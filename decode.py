import random
import math

import numpy as np

alphabet = None
alpha_prior = None
alpha_transition = None

# Read in data from the data files
def populate_globals():
    global alphabet, alpha_prior, alpha_transition

    def read_line(name):
        with open(name) as f:
            return f.readline().split(",")

    def np_load(name):
        with open(name) as f:
            return np.loadtxt(f, delimiter=",")

    alphabet = read_line("data/alphabet.csv")
    alpha_prior = np_load("data/letter_probabilities.csv")
    alpha_transition = np_load("data/letter_transition_matrix.csv")


# Given a permutation, samples (and returns) a new permutation from the proposal distribution.
def sample_proposal_dist(perm):
    N = len(perm)

    # # Simple table to enumerate the unique pairs (x, y) such that x,y are in [0,...,M] (M=N-1)
    # #    0 1 2 - M
    # #    ---------
    # #  M|
    # #  ||
    # #  3|3|4|5
    # #  2|1|2|
    # #  1|0|
    # #  0|
    # #

    # # uniformly randomly choose a pair and extract the two indices
    # pair_idx = random.randint(1, (N * (N - 1)) // 2) - 1
    # x = int(-0.5 + math.sqrt(2 * pair_idx + 0.25))
    # y = pair_idx - ((x * (x + 1)) // 2)
    # x += 1

    [x, y] = random.sample(range(N), 2)

    # construct the new permutation (old with x, y swapped)
    new_perm = [x for x in perm]
    new_perm[x], new_perm[y] = new_perm[y], new_perm[x]

    return new_perm


# Convert the text to indices for easier processing
def to_indices(text):
    pass


# Convert indices back to text
def fron_indices(ids):
    pass


def decode(ciphertext, has_breakpoint):
    populate_globals()

    plaintext = ciphertext  # Replace with your code
    return plaintext
