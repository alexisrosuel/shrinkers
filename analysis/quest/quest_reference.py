"""Faithful NumPy port of the official QuEST function (Ledoit & Wolf 2016).

Reference: O. Ledoit and M. Wolf, "Numerical Implementation of the QuEST
Function", arXiv:1601.05870, and the authors' MATLAB code ``QuEST.m``
(copied next to this file).

QuEST is the *forward* deterministic map

    tau (population eigenvalues)  ->  lambda (limiting sample eigenvalues)

of the Marchenko-Pastur / Silverstein theory.  Exactly as in the reference
code it is computed in ``u = -1 / m_Fbar(z)`` space:

    for a real grid point xi = Re(u), solve for y = Im(u) >= 0

        1 - c * sum_k w_k tau_k^2 / ((tau_k - xi)^2 + y^2) = 0,

    then set u = xi + i y and

        z = u - c u m_LF(u),      m_LF(u) = sum_k w_k tau_k / (tau_k - u)
        f(z) = Im(u) / (pi c |u|^2)                  (sample density)
        d(z) = z / |1 - c m_LF(u)|^2                 (optimal shrinkage)

The sample eigenvalues are the p-bin averages of the generalized inverse
CDF (eq. (3)-(4) of the paper):  q_i = p * int_{(i-1)/p}^{i/p} F^{-1}(v) dv.

Only the "all tau > 0, c <= 1" branch is implemented (no atom at zero),
which covers every spectrum used in the comparison.  Support detection
follows ``supfun07``: support in xi-space is ``{xi : phi(xi) > 1/c}`` with
``phi(xi) = sum_k w_k tau_k^2 / (tau_k - xi)^2``; gaps (spectral
separation) appear between consecutive distinct population eigenvalues when
the minimum of phi on that gap drops below 1/c.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# small numeric helpers
# --------------------------------------------------------------------------
def _bisect(f, lo: float, hi: float, iters: int = 200) -> float:
    """Bisection tolerant of +inf at the ``lo`` end (poles)."""
    flo = f(lo)
    fhi = f(hi)
    if flo > 0.0 >= fhi:
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            fmid = f(mid)
            if fmid > 0.0:
                lo, flo = mid, fmid
            else:
                hi, fhi = mid, fmid
        return 0.5 * (lo + hi)
    # increasing case
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if fmid < 0.0:
            lo, flo = mid, fmid
        else:
            hi, fhi = mid, fmid
    return 0.5 * (lo + hi)


def _distinct_nonzero(tau: np.ndarray):
    t, counts = np.unique(tau, return_counts=True)
    w = counts / float(tau.size)
    zero_weight = float(w[t == 0].sum()) if np.any(t == 0) else 0.0
    keep = t > 0
    return t[keep], w[keep], zero_weight


def _phi(x, t, w):
    d = t - x
    return np.sum(w * t * t / (d * d))


def _support_intervals(t, w, c):
    """Support intervals in xi-space plus the mass (fraction of p) in each."""
    K = t.size
    inv_c = 1.0 / c

    def psi(x):
        return float(_phi(x, t, w) - inv_c)

    expect2 = float(np.sum(w * t * t))
    leftmost = _bisect(psi, t[0] - np.sqrt(c * expect2) - 1.0,
                       t[0] - np.sqrt(c * w[0] * t[0] ** 2) / 2.0)
    rightmost = _bisect(psi, t[-1] + np.sqrt(c * w[-1] * t[-1] ** 2) / 2.0,
                        t[-1] + np.sqrt(c * expect2) + 1.0)

    gaps = []
    for k in range(K - 1):
        lo, hi = t[k], t[k + 1]
        span = hi - lo
        eps = max(span * 1e-10, np.finfo(float).eps * max(abs(lo), abs(hi), 1.0))
        a, b = lo + eps, hi - eps
        gr = (np.sqrt(5.0) - 1.0) / 2.0
        x1, x2 = b - gr * (b - a), a + gr * (b - a)
        f1, f2 = psi(x1), psi(x2)
        for _ in range(80):
            if f1 < f2:
                b, x2, f2 = x2, x1, f1
                x1 = b - gr * (b - a)
                f1 = psi(x1)
            else:
                a, x1, f1 = x1, x2, f2
                x2 = a + gr * (b - a)
                f2 = psi(x2)
        xstar = 0.5 * (a + b)
        if psi(xstar) < 0.0:
            left_root = _bisect(psi, t[k] + eps, xstar)
            right_root = _bisect(psi, xstar, t[k + 1] - eps)
            gaps.append((left_root, right_root, float(np.sum(w[: k + 1]))))

    intervals, counts = [], []
    prev, prev_mass = leftmost, 0.0
    for left_root, right_root, mass_left in gaps:
        intervals.append((prev, left_root))
        counts.append(mass_left - prev_mass)
        prev, prev_mass = right_root, mass_left
    intervals.append((prev, rightmost))
    counts.append(1.0 - prev_mass)
    counts = np.asarray(counts, float)
    return intervals, counts / counts.sum()


def _y_of_xi(xi, t, w, c, iters=64):
    """Im(u) on the support curve at real part xi (vectorized bisection)."""
    xi = np.atleast_1d(np.asarray(xi, float))
    ymax = np.sqrt(c * float(np.sum(w * t * t))) + 1.0
    d2 = (t[None, :] - xi[:, None]) ** 2          # n_xi x K
    kern = c * w * t * t                          # K
    lo = np.zeros(xi.shape)
    hi = np.full(xi.shape, ymax)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        val = np.sum(kern[None, :] / (d2 + mid[:, None] ** 2), axis=1) - 1.0
        pos = val > 0.0
        lo = np.where(pos, mid, lo)
        hi = np.where(pos, hi, mid)
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------
# forward transform
# --------------------------------------------------------------------------
def quest(tau: np.ndarray, n: int, n_grid: int | None = None) -> dict:
    """Compute the QuEST map for population eigenvalues ``tau`` (``c=p/n``)."""
    tau = np.asarray(tau, dtype=float).ravel()
    p = tau.size
    c = p / n
    t, w, zero_weight = _distinct_nonzero(tau)
    if zero_weight > 0.0 or np.any(tau <= 0.0):
        raise NotImplementedError("quest_reference only handles all tau > 0.")

    intervals, counts = _support_intervals(t, w, c)
    if n_grid is None:
        n_grid = max(200, p)
    per_interval = np.maximum(48, np.round(counts * n_grid).astype(int))

    x_all, f_all, F_all, d_all = [], [], [], []
    offset = 0.0
    for iv, ((alo, ahi), npts) in enumerate(zip(intervals, per_interval)):
        theta = np.linspace(0.0, np.pi / 2.0, npts + 2)
        frac = np.sin(theta) ** 2
        xi = alo + (ahi - alo) * frac[1:-1]
        yy = _y_of_xi(xi, t, w, c)
        u = xi + 1j * yy
        m_LF = np.sum((w * t)[None, :] / (t[None, :] - u[:, None]), axis=1)
        z = np.real_if_close(u - c * u * m_LF).real
        dens = yy / (np.pi * c * np.abs(u) ** 2)
        shrink = z / np.abs(1.0 - c * m_LF) ** 2

        # support endpoints (y = 0)
        z_end = np.empty(2)
        for j, xig in enumerate((alo, ahi)):
            m0 = float(np.sum((w * t) / (t - xig)))
            z_end[j] = xig - c * xig * m0

        zg = np.concatenate([[z_end[0]], z, [z_end[1]]])
        fg = np.concatenate([[0.0], dens, [0.0]])
        dg = np.concatenate([[z_end[0]], shrink, [z_end[1]]])
        G = np.concatenate([[0.0], np.cumsum(0.5 * (fg[:-1] + fg[1:]) * np.diff(zg))])
        Fg = offset + G / G[-1] * counts[iv]
        x_all.append(zg)
        f_all.append(fg)
        d_all.append(dg)
        F_all.append(Fg)
        offset = Fg[-1]

    x_all = np.concatenate(x_all)
    f_all = np.concatenate(f_all)
    d_all = np.concatenate(d_all)
    F_all = np.concatenate(F_all)

    lam = _quantiles(F_all, x_all, p)
    d = np.interp(lam, x_all, d_all)
    return {
        "lambda": lam,
        "d": d,
        "x": x_all,
        "f": f_all,
        "F": F_all,
        "support": intervals,
        "c": c,
        "zero_weight": zero_weight,
    }


def _quantiles(F_grid, x_grid, p: int) -> np.ndarray:
    """q_i = p * int over bin i of the generalized inverse CDF.

    The sup convention ``F^{-1}(v)=sup{x: F(x)<=v}`` is implemented by
    keeping the largest ``x`` for every attained CDF level; the integral of
    the (piecewise-linear) inverse is then evaluated exactly per bin.
    """
    F = np.asarray(F_grid, float)
    x = np.asarray(x_grid, float)
    order = np.argsort(F, kind="stable")
    F, x = F[order], x[order]

    # right end of every attained level (collapse flat gap regions)
    levels, start = np.unique(F, return_index=True)
    end = np.append(start[1:], F.size)
    right_x = np.array([x[start[j]:end[j]].max() for j in range(levels.size)])
    # extend to the full [0, 1] domain
    levels = np.concatenate([[0.0], levels, [1.0]])
    right_x = np.concatenate([[right_x[0]], right_x, [right_x[-1]]])
    levels, uniq = np.unique(levels, return_index=True)
    right_x = right_x[uniq]

    def Q(v):
        """Integral of the inverse CDF from 0 to v (v may be an array)."""
        v = np.clip(np.asarray(v, float), 0.0, 1.0)
        j = np.clip(np.searchsorted(levels, v, side="right") - 1, 0, levels.size - 2)
        lj, lj1 = levels[j], levels[j + 1]
        rj, rj1 = right_x[j], right_x[j + 1]
        span = np.where(lj1 > lj, lj1 - lj, 1.0)
        h = v - lj
        seg = h * (rj + 0.5 * h * (rj1 - rj) / span)
        # cumulative integral up to level j
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (right_x[:-1] + right_x[1:]) * np.diff(levels))])
        return cum[j] + seg

    edges = np.arange(p + 1) / p
    return p * (Q(edges[1:]) - Q(edges[:-1]))


def quest_from_c(tau: np.ndarray, c: float, **kw) -> dict:
    """Convenience: specify the concentration ratio ``c`` instead of ``n``."""
    tau = np.asarray(tau, float).ravel()
    n = round(tau.size / c)
    return quest(tau, n, **kw)


# --------------------------------------------------------------------------
# self-test against the Marchenko-Pastur closed form (tau = identity)
# --------------------------------------------------------------------------
def _mp_density(x, c, sigma2=1.0):
    a = sigma2 * (1.0 - np.sqrt(c)) ** 2
    b = sigma2 * (1.0 + np.sqrt(c)) ** 2
    out = np.zeros_like(x)
    m = (x > a) & (x < b)
    out[m] = np.sqrt((x[m] - a) * (b - x[m])) / (2.0 * np.pi * c * sigma2 * x[m])
    return out


if __name__ == "__main__":
    for c in (0.1, 0.25, 0.5, 1.0):
        p = 500
        tau = np.ones(p)
        res = quest(tau, round(p / c))
        x = res["x"]
        f = res["f"]
        ref = _mp_density(x, c)
        mask = ref > 1e-6
        err = np.max(np.abs(f[mask] - ref[mask]) / ref[mask])
        # analytic support
        a, b = (1 - np.sqrt(c)) ** 2, (1 + np.sqrt(c)) ** 2
        sup = res["support"]
        print(f"c={c:>4}: MP density max rel err = {err:.3e} | "
              f"support {sup[0][0]:.6f}..{sup[-1][1]:.6f} (exact {a:.6f}..{b:.6f}) | "
              f"d range {res['d'].min():.6f}..{res['d'].max():.6f} (exact 1.0)")
