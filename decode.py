import numpy as np

# GLOBALS --------------------------------------------------------------------------
ALPHABET = None
ALPHABET_LOG_PROB = None
ALPHABET_LOG_TRANSITION = None
ALPHABET_LOG_TRANSITION_T = None
CIPHER_INDICES = None
CIPHER_TRANSITION_COUNTS = None
CIPHER_TRANSITION_COUNTS_T = None


# Read in data from the data files
def populate_globals(ciphertext):
    global ALPHABET, ALPHABET_LOG_PROB, ALPHABET_LOG_TRANSITION, ALPHABET_LOG_TRANSITION_T, CIPHER_INDICES, CIPHER_TRANSITION_COUNTS, CIPHER_TRANSITION_COUNTS_T

    def read_line(name):
        with open(name) as f:
            return f.readline().strip().split(",")

    def np_load(name):
        with open(name) as f:
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
            ret[CIPHER_INDICES[i], CIPHER_INDICES[i + 1]] += 1
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


# A function that is proportional to the a posteriori distribution of the cipher, given the text
def post(perm):
    global alpha_prior, alpha_transition, cipherindices
    inv_perm = inverse_permutation(perm)
    ret = alpha_prior[inv_perm[cipherindices[0]]]
    for i in range(len(cipherindices) - 1):
        to = inv_perm[cipherindices[i + 1]]
        fro = inv_perm[cipherindices[i]]
        ret *= alpha_transition[to, fro]
    return ret


def acceptance_helper(x, y, inv_x, inv_y):
    global ALPHABET_LOG_TRANSITION, ALPHABET_LOG_TRANSITION_T, CIPHER_TRANSITION_COUNTS, CIPHER_TRANSITION_COUNTS_T
    return (
        np.dot(CIPHER_TRANSITION_COUNTS[x], ALPHABET_LOG_TRANSITION[inv_x])
        + np.dot(CIPHER_TRANSITION_COUNTS_T[x], ALPHABET_LOG_TRANSITION_T[inv_x])
        + np.dot(CIPHER_TRANSITION_COUNTS[y], ALPHABET_LOG_TRANSITION[inv_y])
        + np.dot(CIPHER_TRANSITION_COUNTS_T[y], ALPHABET_LOG_TRANSITION_T[inv_y])
        - (CIPHER_TRANSITION_COUNTS[x, x] * ALPHABET_LOG_TRANSITION[inv_x, inv_x])
        - (CIPHER_TRANSITION_COUNTS[x, y] * ALPHABET_LOG_TRANSITION[inv_x, inv_y])
        - (CIPHER_TRANSITION_COUNTS[y, y] * ALPHABET_LOG_TRANSITION[inv_y, inv_y])
        - (CIPHER_TRANSITION_COUNTS[y, x] * ALPHABET_LOG_TRANSITION[inv_y, inv_x])
    )


# Return a(perm_cand | perm)
def acceptance_probability(pair, inv_perm):
    global ALPHABET_LOG_PROB
    x, y = pair

    cand_el0 = inv_perm[CIPHER_INDICES[0]]
    if CIPHER_INDICES[0] == x:
        cand_el0 = inv_perm[y]
    elif CIPHER_INDICES[0] == y:
        cand_el0 = inv_perm[x]

    numerator = ALPHABET_LOG_PROB[cand_el0] - acceptance_helper(
        x, y, inv_perm[y], inv_perm[x]
    )
    denominator = ALPHABET_LOG_PROB[inv_perm[CIPHER_INDICES[0]]] - acceptance_helper(
        x, y, inv_perm[x], inv_perm[y]
    )

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


def decode(ciphertext, has_breakpoint):
    populate_globals(ciphertext)

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

    # Use the result to decode the text
    print(last_change)
    print(inv_perm)
    plaintext = decode_with_perm(inv_perm)
    return plaintext
