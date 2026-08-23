"""Quantify how badly the far-field cutoff destroys the real part of the
Stieltjes transform, compared to the imaginary part.

Replicates the math in src/stieltjes/cacheblock.rs (windowed method) vs the
exact full sum, for the same test distribution used in the Rust tests
(eigenvalues = ln(1..p), eta = 0.1/sqrt(p)).
"""
import math


def exact(evals, eta):
    p = len(evals)
    reals = [0.0] * p
    imags = [0.0] * p
    for i in range(p):
        li = evals[i]
        for j in range(p):
            d = li - evals[j]
            denom = d * d + eta * eta
            reals[i] += d / denom
            imags[i] += eta / denom
    return reals, imags


def windowed(evals, eta, cut):
    p = len(evals)
    window = cut * eta
    reals = [0.0] * p
    imags = [0.0] * p
    for i in range(p):
        li = evals[i]
        for j in range(p):
            d = li - evals[j]
            if abs(d) > window:
                continue
            denom = d * d + eta * eta
            reals[i] += d / denom
            imags[i] += eta / denom
    return reals, imags


def report(p, cut):
    evals = [math.log(i + 1.0) for i in range(p)]
    eta = 0.1 / math.sqrt(p)
    er, ei = exact(evals, eta)
    wr, wi = windowed(evals, eta, cut)

    # relative error per point, using exact magnitude as scale
    max_rel_r = 0.0
    max_rel_i = 0.0
    mean_rel_r = 0.0
    mean_rel_i = 0.0
    for i in range(p):
        sr = max(abs(er[i]), 1e-12)
        si = max(abs(ei[i]), 1e-12)
        rr = abs(wr[i] - er[i]) / sr
        ri = abs(wi[i] - ei[i]) / si
        max_rel_r = max(max_rel_r, rr)
        max_rel_i = max(max_rel_i, ri)
        mean_rel_r += rr
        mean_rel_i += ri
    mean_rel_r /= p
    mean_rel_i /= p

    # also absolute error relative to imag magnitude (the thing we care about)
    print(f"p={p:6d} cut={cut:5.1f} eta={eta:.5f}")
    print(f"  real:  max_rel={max_rel_r:8.3f}  mean_rel={mean_rel_r:8.3f}")
    print(f"  imag:  max_rel={max_rel_i:8.3f}  mean_rel={mean_rel_i:8.3f}")
    # sample a few interior points
    for i in [p // 4, p // 2, 3 * p // 4]:
        print(f"    pt {i}: exact_real={er[i]:+.4f} win_real={wr[i]:+.4f} "
              f"exact_imag={ei[i]:.4f} win_imag={wi[i]:.4f}")
    print()


for p in [500, 2000, 5000]:
    report(p, 10.0)
