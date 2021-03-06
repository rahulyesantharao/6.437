import numpy as np

# GLOBALS --------------------------------------------------------------------------
ALPHABET = None
ALPHABET_LOG_PROB = None
ALPHABET_LOG_TRANSITION = None
ALPHABET_LOG_TRANSITION_T = None
CIPHER_INDICES = None
CIPHER_TRANSITION_COUNTS = None
CIPHER_TRANSITION_COUNTS_T = None


def print_perm(inv_perm):
    for i in range(len(inv_perm)):
        print(f"{ALPHABET[i]} -> {ALPHABET[inv_perm[i]]}")


# Read in data from the data files
def populate_globals(ciphertext):
    global ALPHABET, ALPHABET_LOG_PROB, ALPHABET_LOG_TRANSITION, ALPHABET_LOG_TRANSITION_T, CIPHER_INDICES, CIPHER_TRANSITION_COUNTS, CIPHER_TRANSITION_COUNTS_T

    def read_line(name):
        with open(name) as f:
            return f.readline().strip().split(",")

    def np_load(name):
        with open(name) as f:
            with np.errstate(divide="ignore"):
                ret = np.log(np.loadtxt(f, delimiter=","))
        ret[np.isneginf(ret)] = -20
        return ret

    def to_indices(text):
        global ALPHABET
        lookup = {a: i for i, a in enumerate(ALPHABET)}
        return tuple(lookup[a] for a in text)

    def count_transitions():
        global ALPHABET, CIPHER_INDICES
        ret = np.zeros((len(ALPHABET), len(ALPHABET)))
        for i in range(len(CIPHER_INDICES) - 1):
            ret[CIPHER_INDICES[i + 1], CIPHER_INDICES[i]] += 1
        return ret

    ALPHABET = read_line("data/alphabet.csv")
    ALPHABET_LOG_PROB = np_load("data/letter_probabilities.csv")
    ALPHABET_LOG_TRANSITION = np_load("data/letter_transition_matrix.csv")
    ALPHABET_LOG_TRANSITION_T = np.transpose(ALPHABET_LOG_TRANSITION)
    CIPHER_INDICES = to_indices(ciphertext)
    CIPHER_TRANSITION_COUNTS = count_transitions()
    CIPHER_TRANSITION_COUNTS_T = np.transpose(CIPHER_TRANSITION_COUNTS)


# Decode the global cipher indices using a given inverse permutation
def decode_with_fn(f):
    global CIPHER_INDICES
    inv_perm = f[0]
    return "".join(ALPHABET[inv_perm[i]] for i in CIPHER_INDICES)


# Metropolis-Hastings --------------------------------------------------------------------------
# Returns a pair of indices that can be swapped in the permutation (proposal step)
def generate_proposal():
    return np.random.choice(len(ALPHABET), 2, replace=False)


# Applies the permutation given by [pair] to [inv_perm] (in place)
def apply_proposal(inv_perm, pair, num):
    x, y = pair
    inv_perm = inv_perm[0]
    inv_perm[x], inv_perm[y] = inv_perm[y], inv_perm[x]


def acceptance_helper(x, y, inv_perm):
    # Can probably be optimized with numpy tricks (e.g. concatenate arrays and only do one np.dot)
    global ALPHABET_LOG_TRANSITION, ALPHABET_LOG_TRANSITION_T, CIPHER_TRANSITION_COUNTS, CIPHER_TRANSITION_COUNTS_T
    inv_x = inv_perm[x]
    inv_y = inv_perm[y]
    return (
        np.dot(CIPHER_TRANSITION_COUNTS[x], ALPHABET_LOG_TRANSITION[inv_x][inv_perm])
        + np.dot(
            CIPHER_TRANSITION_COUNTS_T[x], ALPHABET_LOG_TRANSITION_T[inv_x][inv_perm]
        )
        + np.dot(CIPHER_TRANSITION_COUNTS[y], ALPHABET_LOG_TRANSITION[inv_y][inv_perm])
        + np.dot(
            CIPHER_TRANSITION_COUNTS_T[y], ALPHABET_LOG_TRANSITION_T[inv_y][inv_perm]
        )
        - (CIPHER_TRANSITION_COUNTS[x, x] * ALPHABET_LOG_TRANSITION[inv_x, inv_x])
        - (CIPHER_TRANSITION_COUNTS[y, x] * ALPHABET_LOG_TRANSITION[inv_y, inv_x])
        - (CIPHER_TRANSITION_COUNTS[y, y] * ALPHABET_LOG_TRANSITION[inv_y, inv_y])
        - (CIPHER_TRANSITION_COUNTS[x, y] * ALPHABET_LOG_TRANSITION[inv_x, inv_y])
    )


# Return a(perm_cand | perm)
def acceptance_probability(pair, inv_perm, num):
    global ALPHABET_LOG_PROB
    x, y = pair

    apply_proposal(inv_perm, pair, num)
    numerator = ALPHABET_LOG_PROB[inv_perm[0][CIPHER_INDICES[0]]]
    numerator += acceptance_helper(x, y, inv_perm[0])

    apply_proposal(inv_perm, pair, num)
    denominator = ALPHABET_LOG_PROB[inv_perm[0][CIPHER_INDICES[0]]]
    denominator += acceptance_helper(x, y, inv_perm[0])

    return numerator - denominator


def initial_guess():
    return [np.random.permutation(len(ALPHABET))]


def log_likelihood(inv_perm):
    inv_perm = inv_perm[0]
    ret = ALPHABET_LOG_PROB[inv_perm[CIPHER_INDICES[0]]]
    for i in range(ALPHABET_LOG_TRANSITION.shape[0]):
        ret += np.dot(
            CIPHER_TRANSITION_COUNTS[i], ALPHABET_LOG_TRANSITION[inv_perm[i]][inv_perm]
        )
    return ret
