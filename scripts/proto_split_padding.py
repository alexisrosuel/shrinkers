"""
Prototype: split real/imaginary FFT with DIFFERENT padding & grid resolution.

The imaginary part (Lorentzian, 1/d^2) is short-range -> small padding, coarse
grid is fine. The real part (Hilbert, 1/d) is long-range -> needs big padding,
but it is SMOOTH so a coarser grid may still resolve it.

Current code forces BOTH parts onto the same 65536-point grid (p=1000). This
prototype measures the error of using different (pad, dx) per part, to see if
we can shrink the grid without wrecking accuracy.

Reference: exact O(p^2) Stieltjes.
"""
import numpy as np


def generate_mp_spectrum(p, c, seed=42):
    rng = np.random.default_rng(seed)
    lam_min = max(1.0 - np.sqrt(c), 0.01) ** 2
    lam_max = (1.0 + np.sqrt(c)) ** 2
    u = rng.uniform(0, 1, p)
    t = lam_min + u * (lam_max - lam_min)
    evals = t + rng.uniform(0, 0.1, p)
    evals.sort()
    return evals


def exact_stieltjes(evals, eta):
    p = len(evals)
    diff = evals[:, None] - evals[None, :]
    denom = diff * diff + eta * eta
    re = np.sum(diff / denom, axis=1)
    im = np.sum(eta / denom, axis=1)
    return re, im


def fft_conv(evals, eta, pad, dx):
    """Single-part convolution via FFT. Returns (real, imag) at the eigenvalues."""
    p = len(evals)
    lo_raw, hi_raw = evals[0], evals[-1]
    raw_range = hi_raw - lo_raw
    lo = lo_raw - pad
    hi = hi_raw + pad
    rng = hi - lo
    m = int(np.ceil(rng / dx))
    m = 1 << (m - 1).bit_length()  # next pow2
    dx = rng / m
    half = m // 2

    # density splat
    density = np.zeros(m)
    pos = (evals - lo) / dx
    idx = pos.astype(int)
    frac = pos - idx
    np.add.at(density, idx, 1.0 - frac)
    np.add.at(density, np.minimum(idx + 1, m - 1), frac)

    # kernel (even + i*odd)
    i = np.arange(m)
    signed = np.where(i <= half, i, i - m).astype(float)
    x = signed * dx
    denom = x * x + eta * eta
    k_even = eta / denom
    k_odd = x / denom
    packed = k_even + 1j * k_odd

    dens_freq = np.fft.fft(density)
    kern_freq = np.fft.fft(packed)

    # unpack even/odd spectra
    ck = kern_freq
    cnk = kern_freq[(np.arange(m) - m) % m]
    ke = 0.5 * (ck + np.conj(cnk))
    ko = -0.5j * (ck - np.conj(cnk))
    d = dens_freq
    im_hat = d * ke
    re_hat = d * ko
    packed_out = (im_hat.real - re_hat.imag) + 1j * (im_hat.imag + re_hat.real)
    out = np.fft.ifft(packed_out)

    # interpolate at eigenvalues
    qpos = (evals - lo) / dx
    qidx = qpos.astype(int)
    qfrac = qpos - qidx
    qidx = np.clip(qidx, 0, m - 2)
    g0 = out[qidx]
    g1 = out[qidx + 1]
    re = (g0.imag * (1 - qfrac) + g1.imag * qfrac).real
    im = (g0.real * (1 - qfrac) + g1.real * qfrac).real
    return re, im


def rel_err(a, b, scale):
    return np.max(np.abs(a - b) / (np.abs(scale) + 1e-12))


def max_rel_err(a, b):
    """Error relative to the max magnitude of b (fair for near-cancelling signals)."""
    scale = np.max(np.abs(b))
    return np.max(np.abs(a - b) / (scale + 1e-12))


def main():
    p = 1000
    c = 0.5
    eta = 0.1 / np.sqrt(p)
    evals = generate_mp_spectrum(p, c)
    re_ex, im_ex = exact_stieltjes(evals, eta)

    raw_range = evals[-1] - evals[0]
    print(f"p={p} eta={eta:.5f} raw_range={raw_range:.3f}")
    print(f"current: pad={max(1000*eta, 2*raw_range):.3f} dx=eta/8={eta/8:.6f} "
          f"-> m={1 << (int(np.ceil((2*max(1000*eta,2*raw_range)+raw_range)/(eta/8)))-1).bit_length()}")
    print()

    # Try different (pad, dx) combos for the FULL transform (both parts together)
    print("=== FULL transform (both parts, same grid) ===")
    print(f"{'pad':>8} {'dx':>10} {'m':>7} {'re_err':>10} {'im_err':>10}")
    for pad in [2 * raw_range, 1.5 * raw_range, raw_range, 1000 * eta, 500 * eta]:
        for dx_mult in [8, 16, 32]:
            dx = eta / dx_mult
            re, im = fft_conv(evals, eta, pad, dx)
            m = 1 << (int(np.ceil((2 * pad + raw_range) / dx)) - 1).bit_length()
            print(f"{pad:8.3f} {dx:10.6f} {m:7d} {max_rel_err(re, re_ex):10.4f} "
                  f"{max_rel_err(im, im_ex):10.4f}")
    print()

    # Now: can we use a COARSER grid for the real part specifically?
    # The real part is smooth (Hilbert kernel), so test coarse dx for it.
    print("=== Real part alone: coarse grid (smooth kernel) ===")
    print(f"{'pad':>8} {'dx':>10} {'m':>7} {'re_err':>10}")
    for pad in [2 * raw_range, 1.5 * raw_range, raw_range]:
        for dx_mult in [8, 16, 32, 64]:
            dx = eta / dx_mult
            re, _ = fft_conv(evals, eta, pad, dx)
            m = 1 << (int(np.ceil((2 * pad + raw_range) / dx)) - 1).bit_length()
            print(f"{pad:8.3f} {dx:10.6f} {m:7d} {max_rel_err(re, re_ex):10.4f}")

    print()
    print("=== Padding sensitivity at FIXED dx=eta/8 (the real lever) ===")
    print(f"{'pad':>8} {'m':>7} {'re_err':>10} {'im_err':>10}")
    for pad_mult in [2.0, 1.5, 1.0, 0.75, 0.5, 0.25, 0.1]:
        pad = pad_mult * raw_range
        dx = eta / 8
        re, im = fft_conv(evals, eta, pad, dx)
        m = 1 << (int(np.ceil((2 * pad + raw_range) / dx)) - 1).bit_length()
        print(f"{pad:8.3f} {m:7d} {max_rel_err(re, re_ex):10.4f} {max_rel_err(im, im_ex):10.4f}")

    print()
    print("=== Imag part alone: tiny padding (short-range Lorentzian) ===")
    print(f"{'pad':>8} {'dx':>10} {'m':>7} {'im_err':>10}")
    for pad_mult in [1.0, 0.5, 0.25, 0.1, 0.05]:
        pad = pad_mult * raw_range
        for dx_mult in [8, 16]:
            dx = eta / dx_mult
            _, im = fft_conv(evals, eta, pad, dx)
            m = 1 << (int(np.ceil((2 * pad + raw_range) / dx)) - 1).bit_length()
            print(f"{pad:8.3f} {dx:10.6f} {m:7d} {max_rel_err(im, im_ex):10.4f}")

    print()
    print("=== Robustness across p: shared pad=0.75*range, dx=eta/8 ===")
    print(f"{'p':>6} {'pad_mult':>9} {'m':>7} {'re_err':>10} {'im_err':>10}")
    for pp in [500, 1000, 2000, 5000]:
        e = generate_mp_spectrum(pp, c)
        et = 0.1 / np.sqrt(pp)
        rex, imx = exact_stieltjes(e, et)
        rr = e[-1] - e[0]
        pad = 0.75 * rr
        dx = et / 8
        re, im = fft_conv(e, et, pad, dx)
        m = 1 << (int(np.ceil((2 * pad + rr) / dx)) - 1).bit_length()
        print(f"{pp:6d} {0.75:9.2f} {m:7d} {max_rel_err(re, rex):10.4f} {max_rel_err(im, imx):10.4f}")


if __name__ == "__main__":
    main()
