import numpy as np

# GLOBALS --------------------------------------------------------------------------
ALPHABET = None
ALPHABET_LOG_PROB = None
ALPHABET_LOG_TRANSITION = None
CIPHER_INDICES = None


# Read in data from the data files
def populate_globals(ciphertext):
    global ALPHABET, ALPHABET_LOG_PROB, ALPHABET_LOG_TRANSITION, CIPHER_INDICES

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
        return np.array(list(lookup[a] for a in text))

    ALPHABET = read_line("data/alphabet.csv")
    ALPHABET_LOG_PROB = np_load("data/letter_probabilities.csv")
    ALPHABET_LOG_TRANSITION = np_load("data/letter_transition_matrix.csv")
    CIPHER_INDICES = to_indices(ciphertext)


# Decode the global cipher indices using a given inverse permutation
def decode_with_fn(f):
    global CIPHER_INDICES
    inv_perm1, bp, inv_perm2 = f
    return "".join(ALPHABET[inv_perm1[i]] for i in CIPHER_INDICES[:bp]) + "".join(
        ALPHABET[inv_perm2[i]] for i in CIPHER_INDICES[bp:]
    )


# Metropolis-Hastings --------------------------------------------------------------------------
# Returns transitions to the next state
GAUSSIAN_SPREAD = 20


def generate_proposal():
    return (
        np.random.choice(len(ALPHABET), 2, replace=False),
        int(np.random.normal(scale=GAUSSIAN_SPREAD)),
        np.random.choice(len(ALPHABET), 2, replace=False),
    )


def gaussian(x):
    scalar = 1 / np.sqrt(2 * np.pi * GAUSSIAN_SPREAD ** 2)
    return scalar * (np.exp(-np.power(x, 2) / (2 * np.power(GAUSSIAN_SPREAD, 2))))


# Applies the step given by [step] to [f] (in place)
def apply_proposal(f, step):
    (x1, y1), bp_diff, (x2, y2) = step
    inv_perm1, _, inv_perm2 = f

    inv_perm1[x1], inv_perm1[y1] = inv_perm1[y1], inv_perm1[x1]
    f[1] += bp_diff
    inv_perm2[x2], inv_perm2[y2] = inv_perm2[y2], inv_perm2[x2]


def unapply_proposal(f, step):
    (x1, y1), bp_diff, (x2, y2) = step
    inv_perm1, _, inv_perm2 = f

    inv_perm1[x1], inv_perm1[y1] = inv_perm1[y1], inv_perm1[x1]
    f[1] -= bp_diff
    inv_perm2[x2], inv_perm2[y2] = inv_perm2[y2], inv_perm2[x2]


def acceptance_probability(step, f):
    apply_proposal(f, step)
    numerator = log_likelihood(f)
    unapply_proposal(f, step)
    denominator = log_likelihood(f)
    return numerator - denominator


def initial_guess():
    return [
        np.random.permutation(len(ALPHABET)),
        np.random.randint(1, CIPHER_INDICES.shape[0]),  # CIPHER_INDICES.shape[0] // 2
        np.random.permutation(len(ALPHABET)),
    ]


def log_likelihood(f):
    inv_perm1, bp, inv_perm2 = f

    if bp <= 0 or bp >= CIPHER_INDICES.shape[0]:
        return -np.inf

    ret = ALPHABET_LOG_PROB[inv_perm1[CIPHER_INDICES[0]]]
    # for i in range(bp - 1):
    # ret += ALPHABET_LOG_TRANSITION[
    # inv_perm1[CIPHER_INDICES[i + 1]], inv_perm1[CIPHER_INDICES[i]]
    # ]

    ret += np.sum(
        ALPHABET_LOG_TRANSITION[
            inv_perm1[CIPHER_INDICES[1:bp]], inv_perm1[CIPHER_INDICES[: bp - 1]]
        ]
    )

    ret += ALPHABET_LOG_TRANSITION[
        inv_perm2[CIPHER_INDICES[bp]], inv_perm1[CIPHER_INDICES[bp - 1]]
    ]

    ret += np.sum(
        ALPHABET_LOG_TRANSITION[
            inv_perm2[CIPHER_INDICES[bp + 1 :]], inv_perm2[CIPHER_INDICES[bp:-1]]
        ]
    )
    # for i in range(bp, len(CIPHER_INDICES) - 1):
    # ret += ALPHABET_LOG_TRANSITION[
    # inv_perm2[CIPHER_INDICES[i + 1]], inv_perm2[CIPHER_INDICES[i]]
    # ]

    return ret
