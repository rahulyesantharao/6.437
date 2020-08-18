import numpy as np

d = None

# Run one step of the Metropolis-Hastings algorithm
# Structured like a Gibb's Sampler: updates one field of the state at a time, conditionally
def metropolis_hastings_step(f):
    step = d.generate_proposal()

    changed = False
    for i in range(len(f)):
        a = min(d.acceptance_probability(step, f, i), 0)
        u = np.random.uniform()
        if u < np.exp(a):
            d.apply_proposal(f, step, i)
            changed = True

    return changed


# Full run of metropolis-hastings, to convergence
def metropolis_hastings(has_breakpoint):
    # initial choice
    f = d.initial_guess()

    convergence_len = 2000 if has_breakpoint else 1000

    # Run the MC to convergence
    last_change = 0
    best_f = f
    best_log_likelihood = d.log_likelihood(f)
    for i in range(10000):
        changed = metropolis_hastings_step(f)

        # update the counter for convergence
        if changed:
            last_change = i
            cur_log_likelihood = d.log_likelihood(f)
            if best_log_likelihood is None or cur_log_likelihood > best_log_likelihood:
                best_f = f
                best_log_likelihood = cur_log_likelihood

        # if we've converged, break
        if i - last_change == convergence_len:
            # print("Converged")
            break

    if has_breakpoint:
        return best_f
    else:
        return f


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
        inv_perm = metropolis_hastings(has_breakpoint)
        cur_log_likelihood = d.log_likelihood(inv_perm)
        if best_log_likelihood is None or cur_log_likelihood > best_log_likelihood:
            best_perm = inv_perm
            best_log_likelihood = cur_log_likelihood

    # Use the result to decode the text
    # print(best_perm)
    plaintext = d.decode_with_fn(best_perm)
    return plaintext
