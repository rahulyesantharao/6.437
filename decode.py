import numpy as np

d = None

# Run one step of the Metropolis-Hastings algorithm
def metropolis_hastings_step(inv_perm):
    step = d.generate_proposal()
    a = min(d.acceptance_probability(step, inv_perm), 0)
    u = np.random.uniform()
    if u < np.exp(a):
        d.apply_proposal(inv_perm, step)
        return True
    else:
        return False


# Full run of metropolis-hastings, to convergence
def metropolis_hastings():
    # initial choice
    inv_perm = d.initial_guess()

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


def decode(ciphertext, has_breakpoint):
    # import appropriately
    global d

    if has_breakpoint:
        import decode_breakpoint as de
    else:
        import decode_no_breakpoint as de

    d = de

    # The actual decode logic
    d.populate_globals(ciphertext)

    # 10 independent runs, choose highest likelihood
    best_perm = None
    best_log_likelihood = None
    for i in range(10):
        inv_perm = metropolis_hastings()
        cur_log_likelihood = d.log_likelihood(inv_perm)
        if best_log_likelihood is None or cur_log_likelihood > best_log_likelihood:
            best_perm = inv_perm
            best_log_likelihood = cur_log_likelihood

    # Use the result to decode the text
    plaintext = d.decode_with_perm(best_perm)
    return plaintext
