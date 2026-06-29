"""
initial_state.py
================
Inisialisasi state awal untuk VQE.
Menyiapkan Hartree-Fock reference state sebagai titik awal,
beserta parameter awal ansatz.
"""

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_nature.second_q.mappers import (
    JordanWignerMapper,
    ParityMapper,
    BravyiKitaevMapper,
)
import config


# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────

def _get_mapper(encoding: str):
    enc = encoding.upper()
    n_a = config.N_ACTIVE_ELECTRONS
    if enc == "JW":
        return JordanWignerMapper()
    elif enc == "PARITY":
        return ParityMapper(num_particles=(n_a // 2, n_a // 2))
    elif enc == "BK":
        return BravyiKitaevMapper()
    else:
        raise ValueError(f"Encoding tidak dikenal: {encoding}")


# ──────────────────────────────────────────────────────────────────────────────
# 1. Hartree-Fock reference state
# ──────────────────────────────────────────────────────────────────────────────

def build_hf_state(
    n_active_orbitals: int = None,
    n_active_electrons: int = None,
    encoding: str = None,
) -> QuantumCircuit:
    """
    Bangun QuantumCircuit HF reference state.

    Dalam representasi JW, ini berarti mengisi n_active_electrons qubit
    pertama (spin-up lalu spin-down) dengan X gate.

    Parameters
    ----------
    n_active_orbitals  : jumlah orbital aktif (default dari config)
    n_active_electrons : jumlah elektron aktif (default dari config)
    encoding           : "JW" | "PARITY" | "BK" (default dari config)

    Returns
    -------
    hf_circuit : QuantumCircuit
    """
    if n_active_orbitals is None:
        n_active_orbitals = config.N_ACTIVE_ORBITALS
    if n_active_electrons is None:
        n_active_electrons = config.N_ACTIVE_ELECTRONS
    if encoding is None:
        encoding = config.ENCODING

    mapper = _get_mapper(encoding)

    num_particles = (n_active_electrons // 2, n_active_electrons // 2)

    hf = HartreeFock(
        num_spatial_orbitals=n_active_orbitals,
        num_particles=num_particles,
        qubit_mapper=mapper,
    )

    n_qubits = hf.num_qubits
    print(f"[InitState] HF state: {n_qubits} qubit, "
          f"{n_active_electrons} elektron aktif, encoding={encoding}")
    return hf


# ──────────────────────────────────────────────────────────────────────────────
# 2. Inisialisasi parameter ansatz
# ──────────────────────────────────────────────────────────────────────────────

def init_parameters(
    n_params: int,
    strategy: str = "zeros",
    seed: int = 42,
    scale: float = 0.01,
) -> np.ndarray:
    """
    Inisialisasi vektor parameter ansatz.

    Parameters
    ----------
    n_params : jumlah parameter
    strategy : "zeros"  → semua nol (UCCSD: murni HF di awal)
               "random" → nilai random kecil (berguna untuk menghindari
                          barren plateau pada ansatz acak)
               "random_uniform" → uniform [-π, π]
    seed     : random seed untuk reproduktibilitas
    scale    : skala noise untuk strategi "random"

    Returns
    -------
    x0 : np.ndarray shape (n_params,)
    """
    rng = np.random.default_rng(seed)

    if strategy == "zeros":
        x0 = np.zeros(n_params)

    elif strategy == "random":
        x0 = rng.normal(loc=0.0, scale=scale, size=n_params)

    elif strategy == "random_uniform":
        x0 = rng.uniform(-np.pi, np.pi, size=n_params)

    else:
        raise ValueError(
            f"Strategi inisialisasi '{strategy}' tidak dikenal. "
            "Pilih: zeros | random | random_uniform"
        )

    print(f"[InitState] Parameter awal: {n_params} param, strategi='{strategy}', "
          f"norm={np.linalg.norm(x0):.4f}")
    return x0


# ──────────────────────────────────────────────────────────────────────────────
# 3. State awal kustom (superposisi / referensi lain)
# ──────────────────────────────────────────────────────────────────────────────

def build_custom_state(statevector: np.ndarray) -> QuantumCircuit:
    """
    Inisialisasi state awal dari statevector numerik.
    Berguna untuk restart VQE atau debugging.

    Parameters
    ----------
    statevector : np.ndarray panjang 2^n (ternormalisasi)

    Returns
    -------
    qc : QuantumCircuit
    """
    from qiskit import QuantumCircuit as QC
    n = int(np.log2(len(statevector)))
    assert len(statevector) == 2**n, "Panjang statevector harus pangkat 2."
    norm = np.linalg.norm(statevector)
    assert abs(norm - 1.0) < 1e-8, f"Statevector tidak ternormalisasi (norm={norm:.6f})"

    qc = QC(n)
    qc.initialize(statevector, range(n))
    print(f"[InitState] Custom statevector: {n} qubit")
    return qc


# ──────────────────────────────────────────────────────────────────────────────
# 4. Ekspektasi energi HF (sanity check)
# ──────────────────────────────────────────────────────────────────────────────

def compute_hf_energy(hamiltonian, n_active_orbitals=None, n_active_electrons=None,
                      encoding=None) -> float:
    """
    Hitung ekspektasi energi ⟨HF|H|HF⟩ sebagai upper bound awal VQE.

    Returns
    -------
    e_hf : float (Hartree)
    """
    from qiskit.primitives import StatevectorEstimator

    hf_qc     = build_hf_state(n_active_orbitals, n_active_electrons, encoding)
    estimator = StatevectorEstimator()
    job       = estimator.run([(hf_qc, hamiltonian)])
    e_hf      = job.result()[0].data.evs.real

    print(f"[InitState] ⟨HF|H|HF⟩ = {e_hf:.8f} Ha")
    return float(e_hf)


# ──────────────────────────────────────────────────────────────────────────────
# Test cepat
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    hf = build_hf_state()
    print(hf.draw(output="text", fold=80))

    x0 = init_parameters(n_params=10, strategy="zeros")
    print(f"\nParameter awal (zeros):\n{x0}")

    x0r = init_parameters(n_params=10, strategy="random", scale=0.05)
    print(f"\nParameter awal (random):\n{x0r}")