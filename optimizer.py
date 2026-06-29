"""
optimizer.py
============
Koleksi optimizer untuk VQE:
  - BFGS, L-BFGS-B, COBYLA  → dari SciPy / Qiskit (interface seragam)
  - RMSProp                   → implementasi manual (gradient-based)
  - NFT / RotoSolve           → implementasi manual (gradient-free,
                                 parameter shift pada Pauli rotasi)

Semua fungsi mengembalikan objek OptimizeResult-like dengan atribut:
  .x   → parameter optimal
  .fun → nilai fungsi minimum
  .nit → jumlah iterasi
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional
from scipy.optimize import minimize as scipy_minimize
import config


# ──────────────────────────────────────────────────────────────────────────────
# Dataclass hasil optimizer (kompatibel interface scipy)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class OptResult:
    x      : np.ndarray
    fun    : float
    nit    : int
    nfev   : int
    success: bool  = True
    message: str   = ""
    history: list  = field(default_factory=list)  # (iter, energy) pairs


# ──────────────────────────────────────────────────────────────────────────────
# 1. BFGS
# ──────────────────────────────────────────────────────────────────────────────

def optimize_bfgs(
    cost_fn: Callable,
    x0: np.ndarray,
    tol: float = None,
    maxiter: int = None,
) -> OptResult:
    """
    Optimizer BFGS (Broyden-Fletcher-Goldfarb-Shanno) via SciPy.
    Cocok untuk landscape yang relatif mulus; menggunakan finite-diff gradient.

    Parameters
    ----------
    cost_fn : callable(x) → float
    x0      : parameter awal
    tol     : toleransi konvergensi (default dari config)
    maxiter : maks iterasi (default dari config)

    Returns
    -------
    OptResult
    """
    if tol is None:
        tol = config.OPT_TOL
    if maxiter is None:
        maxiter = config.OPT_MAX_ITER

    history = []
    nfev_counter = [0]

    def tracked(x):
        val = cost_fn(x)
        nfev_counter[0] += 1
        history.append((nfev_counter[0], float(val)))
        return val

    res = scipy_minimize(
        tracked, x0, method="BFGS",
        options={"gtol": tol, "maxiter": maxiter},
    )

    return OptResult(
        x=res.x, fun=res.fun, nit=res.nit,
        nfev=nfev_counter[0], success=res.success,
        message=res.message, history=history,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 2. L-BFGS-B
# ──────────────────────────────────────────────────────────────────────────────

def optimize_lbfgsb(
    cost_fn: Callable,
    x0: np.ndarray,
    tol: float = None,
    maxiter: int = None,
) -> OptResult:
    """
    Optimizer L-BFGS-B (Limited-memory BFGS with box constraints) via SciPy.
    Lebih efisien memori dibanding BFGS untuk masalah berdimensi tinggi.
    """
    if tol is None:
        tol = config.OPT_TOL
    if maxiter is None:
        maxiter = config.OPT_MAX_ITER

    history = []
    nfev_counter = [0]

    def tracked(x):
        val = cost_fn(x)
        nfev_counter[0] += 1
        history.append((nfev_counter[0], float(val)))
        return val

    res = scipy_minimize(
        tracked, x0, method="L-BFGS-B",
        options={"ftol": tol, "gtol": tol, "maxiter": maxiter},
    )

    return OptResult(
        x=res.x, fun=res.fun, nit=res.nit,
        nfev=nfev_counter[0], success=res.success,
        message=res.message, history=history,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 3. COBYLA
# ──────────────────────────────────────────────────────────────────────────────

def optimize_cobyla(
    cost_fn: Callable,
    x0: np.ndarray,
    tol: float = None,
    maxiter: int = None,
) -> OptResult:
    """
    Optimizer COBYLA (Constrained Optimization BY Linear Approximations).
    Gradient-free; andal untuk landscape tidak rata / bising.
    Banyak digunakan dalam VQE.
    """
    if tol is None:
        tol = config.OPT_TOL
    if maxiter is None:
        maxiter = config.OPT_MAX_ITER

    history = []
    nfev_counter = [0]

    def tracked(x):
        val = cost_fn(x)
        nfev_counter[0] += 1
        history.append((nfev_counter[0], float(val)))
        return val

    res = scipy_minimize(
        tracked, x0, method="COBYLA",
        options={"rhobeg": 0.1, "maxiter": maxiter, "catol": tol},
    )

    return OptResult(
        x=res.x, fun=res.fun, nit=res.nfev // max(len(x0), 1),
        nfev=nfev_counter[0], success=res.success,
        message=res.message, history=history,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 4. RMSProp (implementasi manual)
# ──────────────────────────────────────────────────────────────────────────────

def optimize_rmsprop(
    cost_fn: Callable,
    x0: np.ndarray,
    lr: float = None,
    decay: float = None,
    epsilon: float = None,
    maxiter: int = None,
    tol: float = None,
    grad_eps: float = 1e-4,
) -> OptResult:
    """
    RMSProp optimizer (implementasi manual).
    Menggunakan parameter shift rule untuk estimasi gradient kuantum.

    Gradient estimasi via parameter shift:
        ∂E/∂θ_k ≈ [E(θ_k + π/2) - E(θ_k - π/2)] / 2

    Parameters
    ----------
    cost_fn  : callable(x) → float
    x0       : parameter awal
    lr       : learning rate (default config.RMSPROP_LR)
    decay    : decay rate ρ (default config.RMSPROP_DECAY)
    epsilon  : stabilizer numerik (default config.RMSPROP_EPSILON)
    maxiter  : maks iterasi
    tol      : toleransi perubahan energi
    grad_eps : shift untuk finite-difference fallback

    Returns
    -------
    OptResult
    """
    if lr      is None: lr      = config.RMSPROP_LR
    if decay   is None: decay   = config.RMSPROP_DECAY
    if epsilon is None: epsilon = config.RMSPROP_EPSILON
    if maxiter is None: maxiter = config.OPT_MAX_ITER
    if tol     is None: tol     = config.OPT_TOL

    x       = np.array(x0, dtype=float)
    n       = len(x)
    v       = np.zeros(n)        # rata-rata kuadrat gradien
    history = []
    nfev    = 0

    e_prev = cost_fn(x); nfev += 1

    for it in range(maxiter):
        # ── Hitung gradien via parameter shift ────────────────────────────────
        grad = np.zeros(n)
        for k in range(n):
            xp = x.copy(); xp[k] += np.pi / 2
            xm = x.copy(); xm[k] -= np.pi / 2
            ep = cost_fn(xp); nfev += 1
            em = cost_fn(xm); nfev += 1
            grad[k] = (ep - em) / 2.0

        # ── Update RMSProp ─────────────────────────────────────────────────────
        v = decay * v + (1.0 - decay) * grad**2
        x = x - lr * grad / (np.sqrt(v) + epsilon)

        e_curr = cost_fn(x); nfev += 1
        history.append((it + 1, float(e_curr)))

        if (it + 1) % 50 == 0:
            print(f"  [RMSProp iter {it+1:4d}] E = {e_curr:.8f} Ha  "
                  f"|grad| = {np.linalg.norm(grad):.4e}")

        if abs(e_curr - e_prev) < tol:
            print(f"  [RMSProp] Konvergen pada iterasi {it+1}")
            return OptResult(x=x, fun=e_curr, nit=it+1, nfev=nfev,
                             success=True, message="Konvergen", history=history)
        e_prev = e_curr

    return OptResult(x=x, fun=e_prev, nit=maxiter, nfev=nfev,
                     success=False, message="Maks iterasi tercapai", history=history)


# ──────────────────────────────────────────────────────────────────────────────
# 5. NFT / RotoSolve (implementasi manual)
# ──────────────────────────────────────────────────────────────────────────────

def optimize_nft(
    cost_fn: Callable,
    x0: np.ndarray,
    maxiter: int = None,
    tol: float = None,
) -> OptResult:
    """
    Nakanishi-Fujii-Todo (NFT) / RotoSolve optimizer — implementasi manual.

    Algoritma:
    Untuk setiap parameter θ_k secara bergiliran, energi sebagai fungsi θ_k
    berbentuk:  E(θ_k) = A·cos(θ_k) + B·sin(θ_k) + C
    yang dapat diminimalkan secara analitik menggunakan 3 evaluasi energi.

    Minimisasi analitik:
        θ_k* = arctan2(-B, -A) + π  (atau modifikasi agar kovex)

    Referensi:
    - Nakanishi et al., Phys. Rev. Research 2, 043158 (2020)
    - Ostaszewski et al. (RotoSolve), Quantum 5, 391 (2021)

    Parameters
    ----------
    cost_fn : callable(x) → float
    x0      : parameter awal
    maxiter : jumlah sweep penuh (default config.OPT_MAX_ITER)
    tol     : toleransi perubahan energi antar sweep

    Returns
    -------
    OptResult
    """
    if maxiter is None: maxiter = config.OPT_MAX_ITER
    if tol     is None: tol     = config.OPT_TOL

    x       = np.array(x0, dtype=float)
    n       = len(x)
    history = []
    nfev    = 0
    e_prev  = np.inf

    for sweep in range(maxiter):
        for k in range(n):
            # ── 3 evaluasi untuk fit sinusoidal ───────────────────────────────
            # E(θ_k)  = A cos θ_k + B sin θ_k + C
            # Evaluasi di θ_k = 0, π/2, π
            t0, t1, t2 = x[k], x[k] + np.pi/2, x[k] + np.pi

            def e_at(t):
                xt = x.copy(); xt[k] = t
                return cost_fn(xt)

            e0 = e_at(t0); nfev += 1
            e1 = e_at(t1); nfev += 1
            e2 = e_at(t2); nfev += 1

            # Koefisien: A = (e0-e2)/2, B = (e0 + e2)/2 - e1 ? 
            # Fit eksak tiga titik:
            # E(0) = A + C = e0
            # E(π/2) = B + C = e1
            # E(π)  = -A + C = e2
            A = (e0 - e2) / 2.0
            C = (e0 + e2) / 2.0
            B = e1 - C

            # θ* = arctan2(-B, -A) mod 2π  (meminimalkan A cosθ + B sinθ)
            theta_opt = np.arctan2(-B, -A)

            # Periksa apakah θ* + π lebih kecil (minimum bukan maksimum)
            def sinusoid(t):
                return A * np.cos(t - x[k] + t0) + B * np.sin(t - x[k] + t0) + C

            if sinusoid(theta_opt + np.pi) < sinusoid(theta_opt):
                theta_opt += np.pi

            x[k] = theta_opt % (2 * np.pi)

        e_curr = cost_fn(x); nfev += 1
        history.append((sweep + 1, float(e_curr)))

        if (sweep + 1) % 10 == 0:
            print(f"  [NFT sweep {sweep+1:4d}] E = {e_curr:.8f} Ha")

        if abs(e_curr - e_prev) < tol:
            print(f"  [NFT] Konvergen pada sweep {sweep+1}")
            return OptResult(x=x, fun=e_curr, nit=sweep+1, nfev=nfev,
                             success=True, message="Konvergen", history=history)
        e_prev = e_curr

    return OptResult(x=x, fun=e_prev, nit=maxiter, nfev=nfev,
                     success=False, message="Maks iterasi tercapai", history=history)


# ──────────────────────────────────────────────────────────────────────────────
# Dispatcher utama
# ──────────────────────────────────────────────────────────────────────────────

def get_optimizer(optimizer_type: str = None):
    """
    Kembalikan fungsi optimizer(cost_fn, x0) → OptResult.

    Parameters
    ----------
    optimizer_type : "BFGS" | "L-BFGS-B" | "COBYLA" | "RMSProp" | "NFT"
                     (default dari config.OPTIMIZER_TYPE)
    """
    if optimizer_type is None:
        optimizer_type = config.OPTIMIZER_TYPE

    opt = optimizer_type.upper().replace("-", "").replace("_", "")

    dispatch = {
        "BFGS"     : optimize_bfgs,
        "LBFGSB"   : optimize_lbfgsb,
        "COBYLA"   : optimize_cobyla,
        "RMSPROP"  : optimize_rmsprop,
        "NFT"      : optimize_nft,
        "ROTOSOLVE": optimize_nft,   # alias
    }

    if opt not in dispatch:
        raise ValueError(
            f"Optimizer '{optimizer_type}' tidak dikenal. "
            "Pilih: BFGS | L-BFGS-B | COBYLA | RMSProp | NFT"
        )

    fn = dispatch[opt]
    print(f"[Optimizer] Menggunakan: {optimizer_type}")
    return fn


# ──────────────────────────────────────────────────────────────────────────────
# Test cepat
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test semua optimizer pada fungsi sederhana: rosenbrock 2D
    from scipy.optimize import rosen

    x0 = np.array([0.0, 0.0])
    print("=== Test Optimizer pada Rosenbrock 2D ===\n")

    for name in ["BFGS", "L-BFGS-B", "COBYLA", "RMSProp", "NFT"]:
        opt_fn = get_optimizer(name)
        result = opt_fn(rosen, x0.copy(), maxiter=500)
        print(f"  {name:10s}: x*={result.x}, f*={result.fun:.6f}, "
              f"nit={result.nit}, nfev={result.nfev}\n")
