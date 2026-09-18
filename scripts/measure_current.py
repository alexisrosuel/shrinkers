"""
Measure the CURRENT Rust fft5 error vs numpy across p and c.
This is the baseline I must not regress.
"""
import numpy as np
from _common import mp_spectrum, numpy_stieltjes

import shrinkers as rk


def main():
    print(f"{'p':>6} {'c':>4} {'re_err':>10} {'im_err':>10} {'rel%':>8}")
    for p in [200, 500, 1000, 2000, 5000]:
        for c in [0.1, 0.5, 0.9]:
            evals = mp_spectrum(p, c)
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
