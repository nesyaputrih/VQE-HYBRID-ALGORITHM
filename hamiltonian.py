"""
hamiltonian.py
==============
Build molecular Hamiltonian dengan active space (CASCI/CASSCF)
dan konversi ke qubit operator via JW / PARITY / BK.

Pipeline: PySCF → integral h1e/h2e → FermionicOp (manual) → mapper Qiskit Nature.
"""

import numpy as np
from pyscf import gto, scf, mcscf, fci as pyscf_fci, ao2mo
from qiskit_nature.second_q.mappers import (
    JordanWignerMapper,
    ParityMapper,
    BravyiKitaevMapper,
)
from qiskit_nature.second_q.operators import FermionicOp
import config


# ──────────────────────────────────────────────────────────────────────────────
# Helper: mapper selector
# ──────────────────────────────────────────────────────────────────────────────

def get_mapper(encoding: str = None):
    """Kembalikan Qiskit Nature mapper sesuai encoding yang dipilih."""
    if encoding is None:
        encoding = config.ENCODING
    enc = encoding.upper()
    n_a = config.N_ACTIVE_ELECTRONS
    if enc == "JW":
        return JordanWignerMapper()
    elif enc == "PARITY":
        return ParityMapper(num_particles=(n_a // 2, n_a // 2))
    elif enc == "BK":
        return BravyiKitaevMapper()
    else:
        raise ValueError(f"Encoding '{encoding}' tidak dikenal. Pilih: JW | PARITY | BK")


# ──────────────────────────────────────────────────────────────────────────────
# PySCF: CASCI / CASSCF
# ──────────────────────────────────────────────────────────────────────────────

def _run_pyscf_casci(mol: gto.Mole) -> dict:
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()

    ncas  = config.N_ACTIVE_ORBITALS
    nelec = config.N_ACTIVE_ELECTRONS

    mc = mcscf.CASCI(mf, ncas, nelec)
    mc.verbose = 0
    mc.kernel()

    h1e, ecore = mc.get_h1eff()
    h2e = ao2mo.restore(1, mc.get_h2eff(), ncas)

    return {
        "h1e"   : h1e,
        "h2e"   : h2e,
        "ecore" : float(ecore),
        "e_cas" : float(mc.e_tot),
        "mf"    : mf,
        "mc"    : mc,
    }


def _run_pyscf_casscf(mol: gto.Mole) -> dict:
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()

    ncas  = config.N_ACTIVE_ORBITALS
    nelec = config.N_ACTIVE_ELECTRONS

    mc = mcscf.CASSCF(mf, ncas, nelec)
    mc.verbose = 0
    mc.kernel()

    h1e, ecore = mc.get_h1eff()
    h2e = ao2mo.restore(1, mc.get_h2eff(), ncas)

    return {
        "h1e"   : h1e,
        "h2e"   : h2e,
        "ecore" : float(ecore),
        "e_cas" : float(mc.e_tot),
        "mf"    : mf,
        "mc"    : mc,
    }


def _run_pyscf_full(mol: gto.Mole) -> dict:
    """Tanpa active space: pakai semua orbital molekul (full space) dari RHF."""
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()

    nmo = mf.mo_coeff.shape[1]

    hcore = mf.get_hcore()
    h1e = mf.mo_coeff.T @ hcore @ mf.mo_coeff
    h2e = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), nmo)
    ecore = float(mf.energy_nuc())

    return {
        "h1e"   : h1e,
        "h2e"   : h2e,
        "ecore" : ecore,
        "e_cas" : float(mf.e_tot),
        "mf"    : mf,
        "mc"    : None,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Bangun FermionicOp dari integral h1e / h2e
# ──────────────────────────────────────────────────────────────────────────────

def _integrals_to_fermionic_op(h1e: np.ndarray, h2e: np.ndarray,
                                ecore: float, ncas: int) -> FermionicOp:
    """
    [FIX] Konvensi BLOCK ordering: alpha = spin-orbital 0..ncas-1,
    beta = spin-orbital ncas..2*ncas-1. Konsisten dgn konvensi internal
    qiskit_nature.second_q (HartreeFock, UCC/UCCSD).
    """
    n_so = 2 * ncas
    data = {"": ecore}
    for p in range(ncas):
        for q in range(ncas):
            coeff = h1e[p, q]
            if abs(coeff) < 1e-14:
                continue
            key_a = f"+_{p} -_{q}"                 # alpha block
            data[key_a] = data.get(key_a, 0.0) + coeff
            key_b = f"+_{p+ncas} -_{q+ncas}"        # beta block
            data[key_b] = data.get(key_b, 0.0) + coeff
    for p in range(ncas):
        for q in range(ncas):
            for r in range(ncas):
                for s in range(ncas):
                    coeff = 0.5 * h2e[p, q, r, s]
                    if abs(coeff) < 1e-14:
                        continue
                    for s1, s2 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                        pa = p + s1 * ncas
                        qa = q + s1 * ncas
                        rb = r + s2 * ncas
                        sb = s + s2 * ncas
                        key = f"+_{pa} +_{rb} -_{sb} -_{qa}"
                        data[key] = data.get(key, 0.0) + coeff
    data = {k: v for k, v in data.items() if abs(v) > 1e-14}
    return FermionicOp(data, num_spin_orbitals=n_so)

# ──────────────────────────────────────────────────────────────────────────────
# Fungsi utama
# ──────────────────────────────────────────────────────────────────────────────

def build_qubit_hamiltonian(mol: gto.Mole, method: str = None, encoding: str = None):
    """
    Build qubit Hamiltonian dari molekul PySCF.

    Returns
    -------
    qubit_op  : SparsePauliOp (Qiskit)
    cas_data  : dict berisi h1e, h2e, ecore, e_cas, mf, mc
    n_qubits  : jumlah qubit
    """
    if method is None:
        method = config.ACTIVE_SPACE_METHOD
    if encoding is None:
        encoding = config.ENCODING

    ncas  = config.N_ACTIVE_ORBITALS
    nelec = config.N_ACTIVE_ELECTRONS

    # ── 1. PySCF ─────────────────────────────────────────────────────────────
    print(f"[Hamiltonian] {method} CAS({nelec},{ncas}) basis={mol.basis} ...")
    if method.upper() == "CASCI":
        cas_data = _run_pyscf_casci(mol)
    elif method.upper() == "CASSCF":
        cas_data = _run_pyscf_casscf(mol)
    elif method.upper() == "NONE":
        cas_data = _run_pyscf_full(mol)
    else:
        raise ValueError(f"Method '{method}' tidak dikenal. Pilih: CASCI | CASSCF | NONE")

    h1e   = cas_data["h1e"]
    h2e   = cas_data["h2e"]
    ecore = cas_data["ecore"]
    ncas  = h1e.shape[0]  # untuk method="NONE", ini beda dari config.N_ACTIVE_ORBITALS

    print(f"  E_core         = {ecore:.8f} Ha")
    print(f"  E_total ({method}) = {cas_data['e_cas']:.8f} Ha")

    # ── 2. FermionicOp ───────────────────────────────────────────────────────
    fermionic_op = _integrals_to_fermionic_op(h1e, h2e, ecore, ncas)
    print(f"  FermionicOp suku: {len(fermionic_op)}")

    # ── 3. Map ke qubit ───────────────────────────────────────────────────────
    print(f"  Encoding: {encoding}")
    mapper   = get_mapper(encoding)
    qubit_op = mapper.map(fermionic_op)
    # Jangan simplify dengan atol terlalu besar — gunakan threshold kecil
    qubit_op = qubit_op.simplify(atol=1e-12)

    n_qubits = qubit_op.num_qubits
    print(f"  Jumlah qubit : {n_qubits}")
    print(f"  Jumlah suku  : {len(qubit_op)}")

    cas_data["n_qubits"] = n_qubits
    return qubit_op, cas_data, n_qubits


# ──────────────────────────────────────────────────────────────────────────────
# Referensi FCI
# ──────────────────────────────────────────────────────────────────────────────

def compute_fci_energy(mol: gto.Mole) -> float:
    """Hitung energi FCI menggunakan PySCF sebagai referensi eksak."""
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()

    cisolver = pyscf_fci.FCI(mf)
    cisolver.verbose = 0
    e_fci, _ = cisolver.kernel()
    return float(e_fci)


# ──────────────────────────────────────────────────────────────────────────────
# Test cepat
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from molecule import build_mol

    mol = build_mol("H2O", 0.96)
    qubit_op, cas_data, n_q = build_qubit_hamiltonian(mol)
    print(f"\nHamiltonian ({config.ENCODING}): {n_q} qubit, {len(qubit_op)} suku")

    if config.RUN_FCI_REFERENCE:
        e_fci = compute_fci_energy(mol)
        print(f"E_FCI = {e_fci:.8f} Ha")