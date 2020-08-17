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
def decode_with_perm(inv_perm):
    global CIPHER_INDICES
    return "".join(ALPHABET[inv_perm[i]] for i in CIPHER_INDICES)


# Metropolis-Hastings --------------------------------------------------------------------------
# Returns a pair of indices that can be swapped in the permutation (proposal step)
def generate_proposal():
    return np.random.choice(len(ALPHABET), 2, replace=False)


# Applies the permutation given by [pair] to [inv_perm] (in place)
def apply_proposal(inv_perm, pair):
    x, y = pair
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
def acceptance_probability(pair, inv_perm):
    global ALPHABET_LOG_PROB
    x, y = pair

    apply_proposal(inv_perm, pair)
    numerator = ALPHABET_LOG_PROB[inv_perm[CIPHER_INDICES[0]]]
    numerator += acceptance_helper(x, y, inv_perm)

    apply_proposal(inv_perm, pair)
    denominator = ALPHABET_LOG_PROB[inv_perm[CIPHER_INDICES[0]]]
    denominator += acceptance_helper(x, y, inv_perm)

    return numerator - denominator


# Run one step of the Metropolis-Hastings algorithm
def metropolis_hastings_step(inv_perm):
    pair = generate_proposal()
    a = min(acceptance_probability(pair, inv_perm), 0)
    u = np.random.uniform()
    if u < np.exp(a):
        apply_proposal(inv_perm, pair)
        return True
    else:
        return False


# Full run of metropolis-hastings, to convergence
def metropolis_hastings():
    # initial choice
    inv_perm = np.random.permutation(len(ALPHABET))

    # Run the MC to convergence
    last_change = 0
    for i in range(10000):
        changed = metropolis_hastings_step(inv_perm)

        # update the counter for convergence
        if changed:
            last_change = i

        # if we've converged, break
        if i - last_change == 1000:
            break
    return inv_perm


def log_likelihood(inv_perm):
    ret = ALPHABET_LOG_PROB[inv_perm[CIPHER_INDICES[0]]]
    for i in range(ALPHABET_LOG_TRANSITION.shape[0]):
        ret += np.dot(
            CIPHER_TRANSITION_COUNTS[i], ALPHABET_LOG_TRANSITION[inv_perm[i]][inv_perm]
        )
    return ret


def decode(ciphertext):
    populate_globals(ciphertext)

    best_perm = None
    best_log_likelihood = None
    for i in range(10):
        inv_perm = metropolis_hastings()
        cur_log_likelihood = log_likelihood(inv_perm)
        if best_log_likelihood is None or cur_log_likelihood > best_log_likelihood:
            best_perm = inv_perm
            best_log_likelihood = cur_log_likelihood

    # print(i, last_change)
    # print_perm(inv_perm)

    # Use the result to decode the text
    plaintext = decode_with_perm(best_perm)
    return plaintext
