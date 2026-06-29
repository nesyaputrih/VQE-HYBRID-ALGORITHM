"""
vqe_runner.py
=============
Modul utama VQE: memanggil hamiltonian, ansatz, initial_state, dan optimizer,
lalu menjalankan loop VQE untuk satu atau seluruh titik PES (scan ikatan).
Hasil disimpan ke folder results/.
"""

import os
import time
import json
import numpy as np
from pathlib import Path

from qiskit.primitives import StatevectorEstimator

import config
from molecule      import build_mol, scan_bond_lengths
from hamiltonian   import build_qubit_hamiltonian, compute_fci_energy
from ansatz        import build_ansatz, AdaptVQE
from initial_state import build_hf_state, init_parameters, compute_hf_energy
from optimizer     import get_optimizer


# ──────────────────────────────────────────────────────────────────────────────
# Helper: cost function untuk VQE biasa (non-ADAPT)
# ──────────────────────────────────────────────────────────────────────────────

def _make_cost_fn(ansatz_qc, hamiltonian, estimator):
    """
    Kembalikan fungsi energi ⟨ψ(θ)|H|ψ(θ)⟩ yang bisa dipanggil optimizer.
    """
    def cost(params):
        bound = ansatz_qc.assign_parameters(
            dict(zip(ansatz_qc.parameters, params))
        )
        job    = estimator.run([(bound, hamiltonian)])
        energy = job.result()[0].data.evs.real
        return float(energy)
    return cost


# ──────────────────────────────────────────────────────────────────────────────
# VQE untuk satu titik geometri
# ──────────────────────────────────────────────────────────────────────────────

def run_vqe_single(mol, bond_length: float) -> dict:
    """
    Jalankan VQE pada satu titik geometri.

    Parameters
    ----------
    mol         : pyscf.gto.Mole
    bond_length : panjang ikatan (Angstrom) — hanya untuk logging

    Returns
    -------
    result_dict : dict berisi energi VQE, FCI, error, waktu, dll.
    """
    t_start = time.time()
    print(f"\n{'='*60}")
    print(f"  VQE: {config.MOLECULE}  r = {bond_length:.3f} Å")
    print(f"  Method : {config.ACTIVE_SPACE_METHOD} | "
          f"Ansatz : {config.ANSATZ_TYPE} | "
          f"Encoding : {config.ENCODING} | "
          f"Optimizer : {config.OPTIMIZER_TYPE}")
    print(f"{'='*60}")

    # ── 1. Build Hamiltonian ──────────────────────────────────────────────────
    hamiltonian, cas_data, n_qubits = build_qubit_hamiltonian(mol)
    e_core  = cas_data["ecore"]
    e_casXX = cas_data["e_cas"]

    # ── 2. Build Ansatz ───────────────────────────────────────────────────────
    ansatz_obj = build_ansatz(n_qubits, hamiltonian=hamiltonian)

    # ── 3. Inisialisasi Estimator (Statevector exact) ─────────────────────────
    estimator = StatevectorEstimator()

    # ── 4. Energi HF awal ─────────────────────────────────────────────────────
    e_hf = compute_hf_energy(hamiltonian)

    # ── 5. Jalankan optimasi ──────────────────────────────────────────────────
    if isinstance(ansatz_obj, AdaptVQE):
        # ADAPT-VQE: optimizer diteruskan sebagai callable ke .run()
        opt_fn = get_optimizer(config.OPTIMIZER_TYPE)

        def scipy_wrapper(cost_fn, x0_arr):
            return opt_fn(cost_fn, x0_arr)

        e_vqe, final_circuit, adapt_history = ansatz_obj.run(
            estimator=estimator,
            optimizer_fn=scipy_wrapper,
        )
        opt_history = [(i, e) for i, e in enumerate(adapt_history)]
        n_params = len(ansatz_obj.params)

    else:
        # UCCSD / kUpCCGSD: loop optimasi konvensional
        ansatz_qc = ansatz_obj
        n_params  = ansatz_qc.num_parameters

        x0 = init_parameters(n_params, strategy="zeros")
        cost_fn = _make_cost_fn(ansatz_qc, hamiltonian, estimator)
        opt_fn  = get_optimizer(config.OPTIMIZER_TYPE)

        opt_result  = opt_fn(cost_fn, x0)
        e_vqe       = opt_result.fun
        opt_history = opt_result.history
        n_params    = n_params

    # ── 6. Energi FCI referensi ───────────────────────────────────────────────
    e_fci = None
    if config.RUN_FCI_REFERENCE:
        try:
            e_fci = compute_fci_energy(mol)
            error = abs(e_vqe - e_fci)
        except Exception as exc:
            print(f"  [FCI] Gagal: {exc}")
            error = None
    else:
        error = None

    t_elapsed = time.time() - t_start

    # ── 7. Ringkasan ──────────────────────────────────────────────────────────
    print(f"\n  E_HF     = {e_hf:.8f} Ha")
    print(f"  E_{config.ACTIVE_SPACE_METHOD:6s} = {e_casXX:.8f} Ha")
    print(f"  E_VQE    = {e_vqe:.8f} Ha")
    if e_fci is not None:
        print(f"  E_FCI    = {e_fci:.8f} Ha")
        within = "✓" if error <= config.CHEMICAL_ACCURACY else "✗"
        print(f"  Error    = {error:.2e} Ha  [{within} chemical accuracy "
              f"({config.CHEMICAL_ACCURACY:.1e} Ha)]")
    print(f"  Waktu    = {t_elapsed:.2f} s | n_qubit = {n_qubits} | "
          f"n_param = {n_params}")

    return {
        "bond_length"  : bond_length,
        "n_qubits"     : n_qubits,
        "n_params"     : n_params,
        "e_hf"         : float(e_hf),
        "e_cas"        : float(e_casXX),
        "e_vqe"        : float(e_vqe),
        "e_fci"        : float(e_fci) if e_fci is not None else None,
        "error_hartree": float(error) if error is not None else None,
        "chemical_acc" : config.CHEMICAL_ACCURACY,
        "within_chem_acc": (error <= config.CHEMICAL_ACCURACY) if error is not None else None,
        "opt_history"  : opt_history,
        "time_s"       : t_elapsed,
        "config": {
            "molecule"      : config.MOLECULE,
            "method"        : config.ACTIVE_SPACE_METHOD,
            "ansatz"        : config.ANSATZ_TYPE,
            "encoding"      : config.ENCODING,
            "optimizer"     : config.OPTIMIZER_TYPE,
            "basis"         : config.BASIS,
            "n_active_elec" : config.N_ACTIVE_ELECTRONS,
            "n_active_orbs" : config.N_ACTIVE_ORBITALS,
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Scan PES penuh
# ──────────────────────────────────────────────────────────────────────────────

def run_pes_scan() -> list:
    """
    Jalankan VQE pada seluruh panjang ikatan yang didefinisikan di config.
    Simpan hasil ke results/ sebagai JSON.

    Returns
    -------
    all_results : list of dict
    """
    Path(config.RESULTS_DIR).mkdir(exist_ok=True)

    mol_list = scan_bond_lengths(config.MOLECULE, config.BOND_LENGTHS, config.BASIS)
    all_results = []

    print(f"\n[PES Scan] {config.MOLECULE}: {len(mol_list)} titik ikatan")

    for r, mol in mol_list:
        try:
            res = run_vqe_single(mol, r)
            all_results.append(res)
        except Exception as exc:
            print(f"  [ERROR] r={r:.3f} Å: {exc}")
            all_results.append({
                "bond_length": r, "e_vqe": None, "e_fci": None,
                "error_hartree": None, "error": str(exc),
            })

    # ── Simpan JSON ──────────────────────────────────────────────────────────
    prefix  = config.output_prefix()
    outfile = Path(config.RESULTS_DIR) / f"{prefix}_pes.json"

    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n[PES Scan] Selesai. Hasil disimpan: {outfile}")

    # ── Ringkasan tabel ───────────────────────────────────────────────────────
    print(f"\n{'r (Å)':>8} {'E_VQE (Ha)':>14} {'E_FCI (Ha)':>14} "
          f"{'Error (Ha)':>12} {'ChemAcc':>8}")
    print("-" * 60)
    for res in all_results:
        r    = res.get("bond_length", "?")
        evqe = res.get("e_vqe")
        efci = res.get("e_fci")
        err  = res.get("error_hartree")
        ok   = "✓" if res.get("within_chem_acc") else "✗"
        if evqe is not None:
            print(f"{r:>8.3f} {evqe:>14.8f} "
                  f"{efci if efci else '---':>14} "
                  f"{err if err else '---':>12.2e} {ok:>8}")
        else:
            print(f"{r:>8.3f}  ERROR: {res.get('error','')}")

    return all_results


# ──────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = run_pes_scan()
