"""Validate replacing the FFT of the Cauchy kernel with an analytic DFT.

Both `current_fft5` and `analytic_fft` mirror `src/stieltjes/fft5.rs`.
The analytic version skips the kernel forward FFT, computing the even/odd
kernel spectra directly from the periodic Poisson kernel formula:

    K_even[k] = (pi/dx) * exp(-2*pi*eta*|k|/(m*dx))
    K_odd[k]  = -i*pi/dx * sign(k) * exp(-2*pi*eta*|k|/(m*dx))
"""

import numpy as np


def setup(evals, eta):
    p = len(evals)
    lo_raw, hi_raw = evals[0], evals[-1]
    raw = hi_raw - lo_raw
    pad = max(1000.0 * eta, 0.75 * raw)
    lo = lo_raw - pad
    hi = hi_raw + pad
    L = hi - lo
    min_grid = max(8.0 * L / eta, 8 * p)
    m = 1
    while m < min_grid:
        m *= 2
    dx = L / m
    dens = np.zeros(m)
    for lam in evals:
        pos = (lam - lo) / dx
        idx = int(pos)
        frac = pos - idx
        if idx >= m - 1:
            dens[m - 1] += 1
        else:
            dens[idx] += 1 - frac
            dens[idx + 1] += frac
    return lo, dx, m, dens


def interpolate(out, evals, lo, dx, m, invm=True):
    res = np.empty((len(evals), 2))
    for it, q in enumerate(evals):
        pos = (q - lo) / dx
        idx = int(pos)
        frac = pos - idx
        if idx >= m - 1:
            g = out[m - 1]
            res[it] = (g.imag / m, g.real / m)
        else:
            g0 = out[idx]
            g1 = out[idx + 1]
            # out.real holds Im[m_g] channel, out.imag holds Re[m_g] channel
            r = (g0.imag * (1 - frac) + g1.imag * frac) / m
            i = (g0.real * (1 - frac) + g1.real * frac) / m
            res[it] = (r, i)
    return res


def current_fft5(evals, eta):
    lo, dx, m, dens = setup(evals, eta)
    half = m // 2
    sd = np.where(np.arange(m) <= half, np.arange(m), np.arange(m) - m)
    x = sd * dx
    denom = x * x + eta * eta
    packed = eta / denom + 1j * (x / denom)
    D = np.fft.fft(dens)
    K = np.fft.fft(packed)
    ke = np.zeros(m, complex)
    ko = np.zeros(m, complex)
    for k in range(m):
        ck = K[k]
        cnk = K[(m - k) % m]
        ke[k] = 0.5 * (ck + np.conj(cnk))
        ko[k] = -0.5j * (ck - np.conj(cnk))
    prod = D * ke + 1j * (D * ko)  # im_hat + i*re_hat
    out = np.fft.ifft(prod) * m
    return interpolate(out, evals, lo, dx, m), out


def analytic_fft(evals, eta):
    lo, dx, m, dens = setup(evals, eta)
    D = np.fft.fft(dens)
    k = np.arange(m)
    kk = np.minimum(k, m - k)
    r = np.exp(-2 * np.pi * eta / (m * dx))
    ke = (np.pi / dx) * r ** kk
    sign = np.where(kk == 0, 0, np.where(k <= m // 2, 1, -1))
    ko = -1j * (np.pi / dx) * sign * r ** kk
    prod = D * ke + 1j * (D * ko)
    out = np.fft.ifft(prod) * m
    return interpolate(out, evals, lo, dx, m), out


def exact(evals, eta):
    d = evals[None, :] - evals[:, None]
    z = 1.0 / (d - 1j * eta)
    return z.sum(axis=1)


for p, eta in [(1000, 0.05), (1000, 0.5), (5000, 0.01), (2000, 0.2)]:
    evals = np.sort(np.random.default_rng(1).gamma(3, 1, p))
    (cur, _), (ana, _) = current_fft5(evals, eta), analytic_fft(evals, eta)
    ex = exact(evals, eta)

    re_scale = max(np.abs(ex.real)) or 1.0
    im_scale = max(np.abs(ex.imag)) or 1.0
    re_c = np.max(np.abs(cur[:, 0] - ex.real)) / re_scale
    re_a = np.max(np.abs(ana[:, 0] - ex.real)) / re_scale
    im_c = np.max(np.abs(cur[:, 1] - ex.imag)) / im_scale
    im_a = np.max(np.abs(ana[:, 1] - ex.imag)) / im_scale
    d_re = np.max(np.abs(ana[:, 0] - cur[:, 0]))
    d_im = np.max(np.abs(ana[:, 1] - cur[:, 1]))

    print(f"p={p} eta={eta}:")
    print(f"  vs exact Re: current={re_c:.2e}  analytic={re_a:.2e}")
    print(f"  vs exact Im: current={im_c:.2e}  analytic={im_a:.2e}")
    print(f"  analytic vs current max|dRe|={d_re:.2e} max|dIm|={d_im:.2e}")