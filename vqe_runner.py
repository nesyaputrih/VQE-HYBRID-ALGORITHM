"""
vqe_runner.py
=============
Modul utama VQE: memanggil hamiltonian, ansatz, initial_state, dan optimizer,
lalu menjalankan loop VQE untuk satu atau seluruh titik PES (scan ikatan).
Hasil disimpan ke folder output/.
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
    """
    t_total_start = time.time()
    print(f"\n{'='*60}")
    print(f"  VQE: {config.MOLECULE}  r = {bond_length:.3f} Å")
    print(f"  Method : {config.ACTIVE_SPACE_METHOD} | "
          f"Ansatz : {config.ANSATZ_TYPE} | "
          f"Encoding : {config.ENCODING} | "
          f"Optimizer : {config.OPTIMIZER_TYPE}")
    print(f"{'='*60}")

    # ── 1. Build Hamiltonian ──────────────────────────────────────────────────
    t0 = time.time()
    hamiltonian, cas_data, n_qubits = build_qubit_hamiltonian(mol)
    t_ham = time.time() - t0
    e_core  = cas_data["ecore"]
    e_casXX = cas_data["e_cas"]
    print(f"  [Timing] Hamiltonian : {t_ham:.2f} s")

    # ── 2. Build Ansatz ───────────────────────────────────────────────────────
    t0 = time.time()
    ansatz_obj = build_ansatz(n_qubits, hamiltonian=hamiltonian)
    t_ansatz_build = time.time() - t0
    print(f"  [Timing] Build ansatz: {t_ansatz_build:.2f} s")

    # ── 3. Estimator ──────────────────────────────────────────────────────────
    estimator = StatevectorEstimator()

    # ── 4. Energi HF ──────────────────────────────────────────────────────────
    e_hf = compute_hf_energy(hamiltonian)

    # ── 4b. FCI dalam active space (batas atas yang bisa dicapai VQE) ────────────
    # Ini adalah eigenvalue terendah Hamiltonian qubit — batas teoretis VQE
    from qiskit.quantum_info import SparsePauliOp
    import numpy as np
    H_matrix = hamiltonian.to_matrix()
    evals = np.linalg.eigvalsh(H_matrix)
    e_fci_active = float(evals[0].real)   # FCI dalam active space
    print(f"  E_FCI (dalam AS) = {e_fci_active:.8f} Ha  ← batas VQE")

    # FCI penuh via PySCF (untuk perbandingan dengan eksperimen/literatur)
    e_fci = None
    if config.RUN_FCI_REFERENCE:
        try:
            e_fci = compute_fci_energy(mol)
            print(f"  E_FCI (penuh)    = {e_fci:.8f} Ha  ← referensi literatur")
            print(f"  Gap AS vs penuh  = {abs(e_fci_active - e_fci)*1000:.2f} mHa"
                  f"  (tidak bisa dikurangi VQE)")
        except Exception as exc:
            print(f"  [FCI] Gagal: {exc}")

    # ── 5. Optimasi ───────────────────────────────────────────────────────────
    adapt_timing = {}
    t0 = time.time()

    if isinstance(ansatz_obj, AdaptVQE):
        opt_fn = get_optimizer(config.OPTIMIZER_TYPE)

        def scipy_wrapper(cost_fn, x0_arr):
            return opt_fn(cost_fn, x0_arr)

        # Pakai FCI dalam active space sebagai target — ini yang bisa dicapai VQE
        # FCI penuh hanya untuk pelaporan, bukan untuk quality check
        quality_ref = e_fci_active

        e_vqe, final_circuit, adapt_history = ansatz_obj.run(
            estimator=estimator,
            optimizer_fn=scipy_wrapper,
            e_cas_ref=quality_ref,
        )
        opt_history  = [(i, e) for i, e in enumerate(adapt_history)]
        n_params     = len(ansatz_obj.params)
        adapt_timing = getattr(ansatz_obj, "timing_", {})

    else:
        ansatz_qc = ansatz_obj
        n_params  = ansatz_qc.num_parameters
        x0        = init_parameters(n_params, strategy="zeros")
        cost_fn   = _make_cost_fn(ansatz_qc, hamiltonian, estimator)
        opt_fn    = get_optimizer(config.OPTIMIZER_TYPE)
        opt_result  = opt_fn(cost_fn, x0)
        e_vqe       = opt_result.fun
        opt_history = opt_result.history
        adapt_timing = {"n_iter": opt_result.nit, "total_s": time.time() - t0}

    t_opt = time.time() - t0
    print(f"  [Timing] Optimasi    : {t_opt:.2f} s")

    # ── 6. Error ──────────────────────────────────────────────────────────────────
    t0 = time.time()
    error_vs_active = abs(e_vqe - e_fci_active)   # error yg bisa diperbaiki VQE
    error_vs_full   = abs(e_vqe - e_fci) if e_fci is not None else None
    error           = error_vs_active   # dipakai untuk kolom ChemAcc
    t_fci = time.time() - t0

    t_total = time.time() - t_total_start

    # ── 7. Ringkasan ──────────────────────────────────────────────────────────
    def _fmt_t(s):
        if s < 60:   return f"{s:.1f}s"
        if s < 3600: return f"{s/60:.1f}m"
        return f"{s/3600:.2f}h"

    print(f"\n  E_HF              = {e_hf:.8f} Ha")
    print(f"  E_{config.ACTIVE_SPACE_METHOD:6s}          = {e_casXX:.8f} Ha")
    print(f"  E_VQE             = {e_vqe:.8f} Ha")
    print(f"  E_FCI (dlm AS)    = {e_fci_active:.8f} Ha  ← batas VQE")
    w_as = "✓" if error_vs_active <= config.CHEMICAL_ACCURACY else "✗"
    print(f"  Error vs FCI(AS)  = {error_vs_active:.2e} Ha  [{w_as} chem. acc.]")
    if e_fci is not None:
        w_full = "✓" if error_vs_full <= config.CHEMICAL_ACCURACY else "✗"
        print(f"  E_FCI (penuh)     = {e_fci:.8f} Ha  ← referensi literatur")
        print(f"  Error vs FCI(full)= {error_vs_full:.2e} Ha  [{w_full}]")
        print(f"  Gap AS vs full    = {abs(e_fci_active-e_fci)*1000:.2f} mHa"
              f"  (di luar jangkauan VQE)")
    print(f"\n  ┌─ Timing breakdown ──────────────────────────────")
    print(f"  │  Hamiltonian build : {_fmt_t(t_ham)}")
    print(f"  │  Ansatz build      : {_fmt_t(t_ansatz_build)}")
    print(f"  │  Optimasi total    : {_fmt_t(t_opt)}")
    if adapt_timing:
        n_it = adapt_timing.get("n_iter", "?")
        mg   = adapt_timing.get("mean_grad_s", 0)
        mo   = adapt_timing.get("mean_opt_s",  0)
        print(f"  │    └ ADAPT iter   : {n_it}  "
              f"(grad avg {_fmt_t(mg)}, opt avg {_fmt_t(mo)})")
    print(f"  │  FCI reference     : {_fmt_t(t_fci)}")
    print(f"  └─ TOTAL             : {_fmt_t(t_total)}")
    print(f"  n_qubit={n_qubits} | n_param={n_params}")

    return {
        "bond_length"       : bond_length,
        "n_qubits"          : n_qubits,
        "n_params"          : n_params,
        "e_hf"              : float(e_hf),
        "e_cas"             : float(e_casXX),
        "e_vqe"             : float(e_vqe),
        "e_fci_active"      : float(e_fci_active),
        "e_fci"             : float(e_fci) if e_fci is not None else None,
        "error_hartree"     : float(error_vs_active),
        "error_vs_full"     : float(error_vs_full) if error_vs_full is not None else None,
        "chemical_acc"      : config.CHEMICAL_ACCURACY,
        "within_chem_acc"   : error_vs_active <= config.CHEMICAL_ACCURACY,
        "opt_history"    : opt_history,
        "timing": {
            "total_s"        : t_total,
            "hamiltonian_s"  : t_ham,
            "ansatz_build_s" : t_ansatz_build,
            "optimization_s" : t_opt,
            "fci_s"          : t_fci,
            "adapt"          : adapt_timing,
        },
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
    Simpan hasil ke output/ sebagai JSON.
    """
    Path(config.RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    mol_list = scan_bond_lengths(config.MOLECULE, config.BOND_LENGTHS, config.BASIS)
    all_results = []

    t_scan_start = time.time()
    print(f"\n[PES Scan] {config.MOLECULE}: {len(mol_list)} titik ikatan")
    print(f"[PES Scan] Mulai: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    for idx, (r, mol) in enumerate(mol_list):
        print(f"\n[PES Scan] Titik {idx+1}/{len(mol_list)}", flush=True)
        try:
            res = run_vqe_single(mol, r)
            all_results.append(res)
        except Exception as exc:
            import traceback
            print(f"  [ERROR] r={r:.3f} Å: {exc}")
            traceback.print_exc()
            all_results.append({
                "bond_length": r, "e_vqe": None, "e_fci": None,
                "error_hartree": None, "error": str(exc),
                "timing": {"total_s": 0},
            })

    t_scan_total = time.time() - t_scan_start

    # ── Simpan JSON ───────────────────────────────────────────────────────────
    prefix  = config.output_prefix()
    outfile = Path(config.RESULTS_DIR) / f"{prefix}_pes.json"

    with open(outfile, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n[PES Scan] Selesai: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[PES Scan] Hasil disimpan: {outfile}")

    # ── Ringkasan tabel ───────────────────────────────────────────────────────
    def _fmt_t(s):
        if s is None: return "    ---"
        if s < 60:    return f"{s:6.1f}s"
        if s < 3600:  return f"{s/60:6.1f}m"
        return f"{s/3600:6.2f}h"

    hdr = (f"{'r (Å)':>7} {'E_VQE (Ha)':>14} {'E_FCI (Ha)':>14} "
           f"{'Error (Ha)':>11} {'CA':>3} {'n_op':>5} {'t_opt':>8} {'t_total':>8}")
    print(f"\n{hdr}")
    print("─" * len(hdr))

    t_opt_sum   = 0.0
    t_total_sum = 0.0
    for res in all_results:
        r_val  = res.get("bond_length", "?")
        evqe   = res.get("e_vqe")
        efci   = res.get("e_fci")
        err    = res.get("error_hartree")
        ok     = "✓" if res.get("within_chem_acc") else "✗"
        n_op   = res.get("n_params", 0)
        timing = res.get("timing", {})
        t_opt  = timing.get("optimization_s")
        t_tot  = timing.get("total_s")

        t_opt_sum   += t_opt  or 0
        t_total_sum += t_tot  or 0

        if evqe is not None:
            efci_s = f"{efci:.8f}" if efci else "       ---    "
            err_s  = f"{err:.2e}"  if err  else "    ---    "
            print(f"{r_val:>7.3f} {evqe:>14.8f} {efci_s:>14} "
                  f"{err_s:>11} {ok:>3} {n_op:>5} "
                  f"{_fmt_t(t_opt):>8} {_fmt_t(t_tot):>8}")
        else:
            print(f"{r_val:>7.3f}  ERROR: {res.get('error','')[:50]}")

    print("─" * len(hdr))
    print(f"{'TOTAL':>7} {'':>14} {'':>14} {'':>11} {'':>3} {'':>5} "
          f"{_fmt_t(t_opt_sum):>8} {_fmt_t(t_scan_total):>8}")
    print(f"\n[PES Scan] Waktu program keseluruhan : {_fmt_t(t_scan_total)}")
    print(f"[PES Scan] Total waktu optimasi      : {_fmt_t(t_opt_sum)}")

    return all_results


# ──────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = run_pes_scan()