"""Prototype: adaptive Stieltjes transform with balanced real/imag error.

The imaginary part is short-range (Lorentzian, 1/d^2) and truncates cleanly
with a window. The real part is long-range (Hilbert kernel, 1/d) and is
log-divergent — it needs a GLOBAL method.

We test three strategies for the far-field real part:
  A) FFT odd-kernel (reuse fft5 approach)
  B) Treecode/FMM with higher-order multipoles (dipole/quadrupole)
  C) Hybrid: windowed imag + exact far-field real

Goal: choose R (imag window) so imag error ~ real error (balanced).
"""
import numpy as np
from scipy.fft import fft, ifft


def exact_stieltjes(evals, eta):
    """Exact full Stieltjes transform (no cutoff)."""
    p = len(evals)
    re = np.zeros(p)
    im = np.zeros(p)
    for i in range(p):
        d = evals - evals[i]
        denom = d * d + eta * eta
        re[i] = np.sum(d / denom)
        im[i] = np.sum(eta / denom)
    return re, im


def windowed_stieltjes(evals, eta, R):
    """Near-field window: both real & imag within |d|<=R."""
    p = len(evals)
    re = np.zeros(p)
    im = np.zeros(p)
    for i in range(p):
        d = evals - evals[i]
        mask = np.abs(d) <= R
        d = d[mask]
        denom = d * d + eta * eta
        re[i] = np.sum(d / denom)
        im[i] = np.sum(eta / denom)
    return re, im


# ---------- Strategy A: FFT odd-kernel for real part ----------
def fft_real_part(evals, eta):
    p = len(evals)
    lo = evals[0]
    hi = evals[-1]
    raw_range = hi - lo
    pad = max(1000 * eta, 5 * raw_range)
    lo2 = lo - pad
    hi2 = hi + pad
    rng = hi2 - lo2
    m = 1
    while m < max(8 * p, int(np.ceil(8 * rng / eta))):
        m *= 2
    dx = rng / m
    half = m // 2
    dens = np.zeros(m)
    for lam in evals:
        pos = (lam - lo2) / dx
        idx = int(pos)
        frac = pos - idx
        if idx >= m - 1:
            dens[m - 1] += 1
        else:
            dens[idx] += 1 - frac
            dens[idx + 1] += frac
    idxs = np.arange(m)
    signed = np.where(idxs <= half, idxs, idxs - m).astype(float)
    x = signed * dx
    denom = x * x + eta * eta
    kodd = x / denom
    D = fft(dens)
    conv_o = np.real(ifft(D * fft(kodd)))
    re = np.zeros(p)
    for i, lam in enumerate(evals):
        pos = (lam - lo2) / dx
        idx = int(pos)
        frac = pos - idx
        if idx >= m - 1:
            re[i] = conv_o[m - 1]
        else:
            re[i] = conv_o[idx] * (1 - frac) + conv_o[idx + 1] * frac
    return re


# ---------- Strategy B: Treecode with multipoles ----------
def treecode_real_part(evals, theta, order=2):
    """1D treecode for the 1/d kernel with multipole expansion up to `order`.

    For a cluster of points {x_k} with weights, the far-field contribution to
    query point z is sum_k 1/(z - x_k). We expand in multipoles about the
    cluster center mu:
      1/(z - x) = sum_n (x - mu)^n / (z - mu)^(n+1)
    """
    p = len(evals)

    class Node:
        __slots__ = ("cnt", "hi", "left", "lo", "mom", "right")

        def __init__(self, lo, hi, cnt, mom, left, right):
            self.lo = lo
            self.hi = hi
            self.cnt = cnt
            self.mom = mom  # moments: mom[n] = sum_k (x_k - mu)^n
            self.left = left
            self.right = right

    def build(arr, lo, hi):
        if len(arr) == 0:
            return Node(lo, hi, 0, np.zeros(order + 1), None, None)
        if len(arr) == 1:
            mom = np.zeros(order + 1)
            mom[0] = 1.0
            return Node(lo, hi, 1, mom, None, None)
        mid = (lo + hi) / 2
        split = np.searchsorted(arr, mid)
        left = build(arr[:split], lo, mid)
        right = build(arr[split:], mid, hi)
        cnt = left.cnt + right.cnt
        mu = (left.mom[0] * (left.lo + left.hi) / 2 + right.mom[0] * (right.lo + right.hi) / 2) / cnt if cnt else 0
        # recompute moments about mu from children (approx: use child centers)
        mom = np.zeros(order + 1)
        for child in (left, right):
            if child.cnt == 0:
                continue
            cmu = (child.lo + child.hi) / 2
            for n in range(order + 1):
                mom[n] += child.cnt * (cmu - mu) ** n
        return Node(lo, hi, cnt, mom, left, right)

    root = build(evals, evals[0], evals[-1])

    def query(node, z, theta):
        if node.cnt == 0:
            return 0.0
        mu = (node.lo + node.hi) / 2
        if node.left is None and node.right is None:
            return 0.0 if abs(z - mu) < 1e-12 else 1.0 / (z - mu)
        dist = abs(z - mu)
        size = node.hi - node.lo
        if size / dist < theta:
            # multipole expansion about mu
            dz = z - mu
            s = 0.0
            for n in range(order + 1):
                s += node.mom[n] / dz ** (n + 1)
            return s
        return query(node.left, z, theta) + query(node.right, z, theta)

    return np.array([query(root, li, theta) for li in evals])


# ---------- Strategy C: Hybrid (windowed imag + exact far-field real) ----------
def hybrid_stieltjes(evals, eta, R):
    """Near-field window for both parts + exact far-field real."""
    p = len(evals)
    re = np.zeros(p)
    im = np.zeros(p)
    for i in range(p):
        d = evals - evals[i]
        mask = np.abs(d) <= R
        d_near = d[mask]
        denom_near = d_near * d_near + eta * eta
        re[i] = np.sum(d_near / denom_near)
        im[i] = np.sum(eta / denom_near)
        d_far = d[~mask]
        denom_far = d_far * d_far + eta * eta
        re[i] += np.sum(d_far / denom_far)
    return re, im


def rel_err(a, b):
    return np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12))


def main():
    p = 2000
    evals = np.sort(np.log(np.arange(1, p + 1) + 1.0))
    eta = 0.1 / np.sqrt(p)
    full_re, full_im = exact_stieltjes(evals, eta)

    print("=" * 70)
    print("Strategy A: FFT odd-kernel for real part")
    print("=" * 70)
    re_a = fft_real_part(evals, eta)
    print(f"  FFT real rel err: {rel_err(re_a, full_re):.4f}")

    print()
    print("=" * 70)
    print("Strategy B: Treecode with multipoles for real part")
    print("=" * 70)
    for theta in [0.5, 0.3, 0.2]:
        for order in [1, 2, 3]:
            re_b = treecode_real_part(evals, theta, order)
            print(f"  theta={theta} order={order}: rel err {rel_err(re_b, full_re):.4f}")

    print()
    print("=" * 70)
    print("Strategy C: Hybrid (windowed imag + exact far-field real)")
    print("=" * 70)
    for mult in [10, 50, 100]:
        R = mult * eta
        re_c, im_c = hybrid_stieltjes(evals, eta, R)
        print(f"  R/eta={mult}: real err {rel_err(re_c, full_re):.6f}  "
              f"imag err {rel_err(im_c, full_im):.6f}")


if __name__ == "__main__":
    main()
