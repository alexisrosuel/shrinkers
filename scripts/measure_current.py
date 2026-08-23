"""
Measure the CURRENT Rust fft5 error vs numpy across p and c.
This is the baseline I must not regress.
"""
import numpy as np

import shrinkers as rk


def generate_mp_spectrum(p, c=0.5, seed=42):
    rng = np.random.default_rng(seed)
    lm = max(1.0 - np.sqrt(c), 0.01) ** 2
    lM = (1.0 + np.sqrt(c)) ** 2
    u = rng.uniform(0, 1, p)
    t = lm + u * (lM - lm)
    return np.sort(t + rng.uniform(0, 0.1, p))


def numpy_stieltjes(evals, eta):
    diff = evals[:, None] - evals[None, :]
    denom = diff * diff + eta * eta
    return np.mean(diff / denom, axis=1), np.mean(eta / denom, axis=1)


def main():
    print(f"{'p':>6} {'c':>4} {'re_err':>10} {'im_err':>10} {'rel%':>8}")
    for p in [200, 500, 1000, 2000, 5000]:
        for c in [0.1, 0.5, 0.9]:
            evals = generate_mp_spectrum(p, c)
            eta = 0.1 / np.sqrt(p)
            ref_r, ref_i = numpy_stieltjes(evals, eta)
            ref_scale = max(np.max(np.abs(ref_r)), np.max(np.abs(ref_i)))
            res = rk.stieltjes_transform(evals, eta, method="fft2")
            re_err = np.max(np.abs(res["real"] - ref_r))
            im_err = np.max(np.abs(res["imag"] - ref_i))
            rel = max(re_err, im_err) / ref_scale * 100
            print(f"{p:>6} {c:>4} {re_err:>10.3e} {im_err:>10.3e} {rel:>7.3f}%")


if __name__ == "__main__":
    main()
