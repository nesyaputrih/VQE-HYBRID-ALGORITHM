"""
ansatz.py
=========
Build ansatz VQE: UCCSD, kUpCCGSD, dan ADAPT-VQE.
Semua output berupa QuantumCircuit Qiskit yang siap digunakan dalam VQE loop.

[VERSION] 2026-07-01-v5-spin-conserving-pool
Fitur: ADAPT-VQE dengan variational principle sanity check + auto-restart
       jika E_VQE > E_HF (indikasi local minimum / optimizer gagal).
       [FIX v5] Operator pool sekarang difilter agar hanya memuat eksitasi
       yang menjaga konservasi spin (Sz). Sebelumnya pool memuat SEMUA
       kombinasi spin-orbital tanpa filter, termasuk eksitasi spin-flip
       (alpha<->beta) yang tidak fisis untuk molekul singlet seperti H2O.
       Ini diduga menjadi penyebab utama lonjakan error VQE-FCI di region
       disosiatif (R >= 2.2 A), karena ADAPT bisa memilih operator yang
       secara gradien "menarik" tapi mengontaminasi spin state.
"""

import numpy as np
import time
from typing import Optional, Tuple, List

from qiskit.circuit import QuantumCircuit, Parameter, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.circuit.library import (
    UCCSD,
    UCC,
)
from qiskit_nature.second_q.mappers import (
    JordanWignerMapper,
    ParityMapper,
    BravyiKitaevMapper,
)
from qiskit_nature.second_q.circuit.library import HartreeFock

import config


# ──────────────────────────────────────────────────────────────────────────────
# Helper: mapper
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
# 1. UCCSD ansatz
# ──────────────────────────────────────────────────────────────────────────────

def build_uccsd(n_qubits: int, encoding: str = None) -> QuantumCircuit:
    """
    Build ansatz UCCSD via Qiskit Nature.

    Parameters
    ----------
    n_qubits : jumlah qubit dari Hamiltonian
    encoding : "JW" | "PARITY" | "BK"

    Returns
    -------
    ansatz : QuantumCircuit dengan parameter bebas
    """
    if encoding is None:
        encoding = config.ENCODING

    n_elec = config.N_ACTIVE_ELECTRONS
    n_orbs = config.N_ACTIVE_ORBITALS
    mapper  = _get_mapper(encoding)

    num_particles = (n_elec // 2, n_elec // 2)
    num_spatial_orbitals = n_orbs

    hf_init = HartreeFock(
        num_spatial_orbitals=num_spatial_orbitals,
        num_particles=num_particles,
        qubit_mapper=mapper,
    )

    ansatz = UCCSD(
        num_spatial_orbitals=num_spatial_orbitals,
        num_particles=num_particles,
        qubit_mapper=mapper,
        initial_state=hf_init,
    )

    print(f"[UCCSD] Ansatz: {ansatz.num_qubits} qubit, "
          f"{ansatz.num_parameters} parameter")
    return ansatz


# ──────────────────────────────────────────────────────────────────────────────
# 2. kUpCCGSD ansatz
# ──────────────────────────────────────────────────────────────────────────────

def build_kupccgsd(n_qubits: int, k: int = None, encoding: str = None) -> QuantumCircuit:
    """
    Build ansatz k-UpCCGSD (generalized pair doubles, k layer).

    k-UpCCGSD menggunakan UCC dengan excitation = ['gsd'] diulang k kali.

    Parameters
    ----------
    n_qubits : jumlah qubit
    k        : jumlah layer (default dari config.K_UPCCGSD)
    encoding : "JW" | "PARITY" | "BK"

    Returns
    -------
    ansatz : QuantumCircuit
    """
    if k is None:
        k = config.K_UPCCGSD
    if encoding is None:
        encoding = config.ENCODING

    n_elec = config.N_ACTIVE_ELECTRONS
    n_orbs = config.N_ACTIVE_ORBITALS
    mapper  = _get_mapper(encoding)

    num_particles        = (n_elec // 2, n_elec // 2)
    num_spatial_orbitals = n_orbs

    hf_init = HartreeFock(
        num_spatial_orbitals=num_spatial_orbitals,
        num_particles=num_particles,
        qubit_mapper=mapper,
    )

    # Bangun k layer GSD dan tumpuk secara sekuensial
    layers = []
    for i in range(k):
        layer = UCC(
            num_spatial_orbitals=num_spatial_orbitals,
            num_particles=num_particles,
            excitations="gsd",
            qubit_mapper=mapper,
            initial_state=hf_init if i == 0 else None,
        )
        # Beri nama parameter unik per layer
        new_params = ParameterVector(f"θ_k{i}", length=layer.num_parameters)
        param_map  = dict(zip(layer.parameters, new_params))
        layer      = layer.assign_parameters(param_map)
        layers.append(layer)

    if k == 1:
        ansatz = layers[0]
    else:
        ansatz = layers[0]
        for l in layers[1:]:
            ansatz = ansatz.compose(l)

    print(f"[kUpCCGSD k={k}] Ansatz: {ansatz.num_qubits} qubit, "
          f"{ansatz.num_parameters} parameter")
    return ansatz


# ──────────────────────────────────────────────────────────────────────────────
# 3. ADAPT-VQE
# ──────────────────────────────────────────────────────────────────────────────

class AdaptVQE:
    """
    Implementasi ADAPT-VQE.

    Algoritma:
    1. Mulai dari state HF.
    2. Hitung gradien ⟨[H, A_k]⟩ untuk setiap operator A_k dalam pool.
    3. Tambahkan operator dengan gradien terbesar ke ansatz.
    4. Optimalkan semua parameter.
    5. Ulangi hingga max_iter atau semua gradien < grad_tol.

    Referensi: Grimsley et al., Nature Commun. 10, 3007 (2019).
    """

    def __init__(
        self,
        hamiltonian: SparsePauliOp,
        n_qubits: int,
        encoding: str = None,
        max_iter: int = None,
        grad_tol: float = None,
        pool_type: str = None,
    ):
        self.hamiltonian = hamiltonian
        self.n_qubits    = n_qubits
        self.encoding    = encoding  or config.ENCODING
        self.max_iter    = max_iter  or config.ADAPT_MAX_ITER
        self.grad_tol    = grad_tol  or config.ADAPT_GRAD_TOL
        self.pool_type   = pool_type or config.ADAPT_POOL

        self.selected_ops: List[SparsePauliOp] = []
        self.params: List[Parameter]           = []
        self.circuit: Optional[QuantumCircuit]  = None
        self._build_hf_init()
        self._build_operator_pool()

    # ── HF initial state ──────────────────────────────────────────────────────
    def _build_hf_init(self):
        n_elec = config.N_ACTIVE_ELECTRONS
        n_orbs = config.N_ACTIVE_ORBITALS
        mapper  = _get_mapper(self.encoding)
        self.hf_circuit = HartreeFock(
            num_spatial_orbitals=n_orbs,
            num_particles=(n_elec // 2, n_elec // 2),
            qubit_mapper=mapper,
        )

    # ── Operator pool ─────────────────────────────────────────────────────────
    def _build_operator_pool(self):
        """
        Build pool operator ADAPT dari semua kemungkinan single dan double
        excitation pada spin-orbital space (bukan hanya HF occ/virt).

        Untuk menangani karakter multi-reference di geometri stretched,
        semua pasangan (p,q) dan (p,q,r,s) dimasukkan ke pool tanpa
        membatasi pada occupied/virtual dari HF reference.

        [FIX v5] Filter konservasi spin (Sz) ditambahkan: konvensi indeks
        spin-orbital interleaved (genap = alpha, ganjil = beta), sama
        seperti di hamiltonian.py. Tanpa filter ini, pool memuat operator
        spin-flip (alpha<->beta) yang tidak fisis untuk molekul singlet
        closed-shell seperti H2O, dan bisa dipilih ADAPT karena gradiennya
        nonzero secara numerik — terutama berbahaya di region disosiatif
        di mana orbital HOMO/LUMO nyaris degenerate.
        """
        from qiskit.quantum_info import SparsePauliOp as SPS
        from itertools import combinations

        n_so = 2 * config.N_ACTIVE_ORBITALS   # jumlah spin-orbital total
        pool = []
        n_rejected_spin = 0

        # ── Helper: single excitation a†_p a_q - h.c. ────────────────────────
        def _single_exc(p, q):
            if p == q:
                return None
            paulis, coeffs = [], []
            # XY term
            lbl = ["I"] * n_so
            lbl[p] = "X"; lbl[q] = "Y"
            for m in range(min(p,q)+1, max(p,q)):
                lbl[m] = "Z"
            paulis.append("".join(reversed(lbl))); coeffs.append(0.5)
            # YX term
            lbl = ["I"] * n_so
            lbl[p] = "Y"; lbl[q] = "X"
            for m in range(min(p,q)+1, max(p,q)):
                lbl[m] = "Z"
            paulis.append("".join(reversed(lbl))); coeffs.append(-0.5)
            op = SPS(paulis, coeffs=coeffs).simplify(atol=1e-12)
            return op if len(op) > 0 else None

        # ── Helper: double excitation a†_p a†_q a_r a_s - h.c. ──────────────
        def _double_exc(p, q, r, s):
            i, j = sorted([p, q])
            k, l = sorted([r, s])
            if len({i, j, k, l}) < 4:   # semua indeks harus unik
                return None

            pauli_patterns = [
                (("X","Y","X","X"),  0.125),
                (("X","X","X","Y"), -0.125),
                (("Y","X","X","X"), -0.125),
                (("X","X","Y","X"),  0.125),
                (("Y","Y","Y","X"),  0.125),
                (("Y","X","Y","Y"), -0.125),
                (("X","Y","Y","Y"), -0.125),
                (("Y","Y","X","Y"),  0.125),
            ]
            paulis, coeffs = [], []
            for (xi,xj,xk,xl), c in pauli_patterns:
                lbl = ["I"] * n_so
                lbl[i]=xi; lbl[j]=xj; lbl[k]=xk; lbl[l]=xl
                for m in range(i+1, j): lbl[m] = "Z"
                for m in range(j+1, k): lbl[m] = "Z"
                for m in range(k+1, l): lbl[m] = "Z"
                paulis.append("".join(reversed(lbl))); coeffs.append(c)
            op = SPS(paulis, coeffs=coeffs).simplify(atol=1e-12)
            return op if len(op) > 0 else None

        # ── Singles: hanya pasangan spin-orbital dengan spin SAMA ────────────
        # (alpha<->alpha atau beta<->beta). Konvensi: indeks genap = alpha,
        # indeks ganjil = beta (lihat hamiltonian.py::_integrals_to_fermionic_op).
        for p, q in combinations(range(n_so), 2):
            if p % 2 != q % 2:
                n_rejected_spin += 1
                continue
            op = _single_exc(p, q)
            if op is not None:
                pool.append(op)

        # ── Doubles: hanya kuartet yang menjaga total Sz ───────────────────
        # creation pair (p,q) harus punya jumlah paritas spin sama dengan
        # annihilation pair (r,s): Sz(p)+Sz(q) == Sz(r)+Sz(s).
        for p, q, r, s in combinations(range(n_so), 4):
            if (p % 2 + q % 2) != (r % 2 + s % 2):
                n_rejected_spin += 1
                continue
            op = _double_exc(p, q, r, s)
            if op is not None:
                pool.append(op)

        self.pool = pool
        print(f"[ADAPT-VQE] Pool: {len(pool)} operator spin-conserving "
              f"({n_rejected_spin} operator spin-flip ditolak) "
              f"(n_so={n_so}, singles+doubles) [{self.pool_type}]")

    # ── Gradien via parameter shift pada statevector ──────────────────────────
    def _compute_gradients(self, statevector: np.ndarray) -> np.ndarray:
        """
        Hitung |⟨ψ|[H, iA_k]|ψ⟩| untuk setiap A_k dalam pool yang belum dipakai.

        Gunakan statevector langsung (bukan Estimator + sirkuit) agar lebih
        efisien dan menghindari masalah decompose/bind.

        Gradien eksak:
            grad_k = |⟨ψ|[H, iA_k]|ψ⟩|
                   = |⟨ψ|H·(iA_k)|ψ⟩ - ⟨ψ|(iA_k)·H|ψ⟩|

        Implementasi via matriks sparse dari SparsePauliOp.
        """
        from qiskit.quantum_info import Statevector

        psi   = statevector          # np.ndarray shape (2^n,)
        H_mat = self.hamiltonian.to_matrix(sparse=True)
        Hpsi  = H_mat.dot(psi)

        grads = np.zeros(len(self.pool))
        for k, A_k in enumerate(self.pool):
            if k in self._used_indices:
                grads[k] = 0.0
                continue
            try:
                A_mat  = A_k.to_matrix(sparse=True)
                Apsi   = A_mat.dot(psi)
                # ⟨ψ|H·iA|ψ⟩ - ⟨ψ|iA·H|ψ⟩ = i(⟨Hψ|Aψ⟩ - ⟨Aψ|Hψ⟩) ... tapi
                # karena A_k real dan anti-Hermitian, pakai langsung:
                # grad = 2 * Im(⟨ψ|H·A|ψ⟩)
                HAp  = H_mat.dot(Apsi)
                grad = 2.0 * abs(np.vdot(psi, HAp).imag)
                grads[k] = grad
            except Exception:
                grads[k] = 0.0

        return grads

    def _get_statevector(self, param_vals: list) -> np.ndarray:
        """Dapatkan statevector sirkuit saat ini via Qiskit Statevector."""
        from qiskit.quantum_info import Statevector

        if self.selected_ops:
            qc = self._build_circuit_parametric()
            bound = qc.assign_parameters(dict(zip(self.params, param_vals)))
        else:
            bound = self.hf_circuit.copy()

        sv = Statevector(bound)
        return sv.data  # np.ndarray complex128

    # ── Build sirkuit parametrik dari operator terpilih ───────────────────────
    def _build_circuit_parametric(self) -> QuantumCircuit:
        """Susun QuantumCircuit parametrik: HF init + operator ADAPT terpilih."""
        from qiskit.circuit.library import PauliEvolutionGate
        from qiskit.synthesis import LieTrotter

        qc = self.hf_circuit.copy()
        for op, param in zip(self.selected_ops, self.params):
            evo = PauliEvolutionGate(op, time=param, synthesis=LieTrotter(reps=1))
            qc.append(evo, range(self.n_qubits))
        return qc.decompose(reps=2)

    # ── Run ADAPT ─────────────────────────────────────────────────────────────
    def run(self, estimator, optimizer_fn, e_cas_ref: float = None
            ) -> Tuple[float, QuantumCircuit, list]:
        """
        Jalankan loop ADAPT-VQE (Grimsley et al. 2019).

        Parameters
        ----------
        e_cas_ref : energi FCI (atau CASCI sebagai fallback) sebagai referensi
                    untuk mendeteksi konvergensi prematur. Jika E_VQE jauh
                    di atas e_cas_ref (gap > ADAPT_QUALITY_GAP_TOL), dianggap
                    belum konvergen dan akan di-restart dengan perturbasi.
        """
        from qiskit.primitives import StatevectorEstimator
        from qiskit.quantum_info import Statevector

        est = StatevectorEstimator()

        # ── E_HF sebagai sanity bound (variational principle: E_VQE ≤ E_HF) ────
        sv_hf = Statevector(self.hf_circuit)
        e_hf_ref = float(np.real(
            sv_hf.expectation_value(self.hamiltonian)
        ))

        max_restarts   = getattr(config, "ADAPT_MAX_RESTARTS", 5)
        quality_gap_tol = getattr(config, "ADAPT_QUALITY_GAP_TOL", 0.05)  # 50 mHa
        rng = np.random.default_rng(42)

        best_e, best_circuit, best_energies, best_timing = None, None, None, None

        for attempt in range(max_restarts + 1):
            e_final, circuit, energies, timing = self._run_single_pass(
                estimator=est,
                optimizer_fn=optimizer_fn,
                perturb_scale=0.0 if attempt == 0 else 0.3 * attempt,
                rng=rng,
            )

            # ── Simpan hasil terbaik sejauh ini ────────────────────────────────
            if best_e is None or (not np.isnan(e_final) and e_final < best_e):
                best_e, best_circuit = e_final, circuit
                best_energies, best_timing = energies, timing

            # ── Sanity check 1: variational principle (E_VQE ≤ E_HF) ──────────
            violates_variational = (not np.isnan(e_final)) and (e_final > e_hf_ref + 1e-6)

            # ── Sanity check 2: gap terhadap CASCI/CASSCF referensi ────────────
            violates_quality = False
            if e_cas_ref is not None and not np.isnan(e_final):
                gap = e_final - e_cas_ref
                violates_quality = gap > quality_gap_tol
                if attempt == 0 or violates_quality:
                    print(f"  [ADAPT-VQE] Gap vs referensi (FCI): {gap*1000:.2f} mHa "
                          f"(toleransi: {quality_gap_tol*1000:.0f} mHa)")

            if not violates_variational and not violates_quality:
                self.timing_ = timing
                self.circuit = circuit
                return e_final, circuit, energies

            reason = []
            if violates_variational:
                reason.append(f"E_VQE({e_final:.6f}) > E_HF({e_hf_ref:.6f})")
            if violates_quality:
                reason.append(f"gap CASCI {gap*1000:.1f} mHa > {quality_gap_tol*1000:.0f} mHa")
            print(f"  [ADAPT-VQE] ⚠ Belum konvergen baik [{', '.join(reason)}] "
                  f"— Retry {attempt+1}/{max_restarts} dengan perturbasi parameter ...")

        # Jika semua restart gagal, kembalikan hasil TERBAIK yang pernah didapat
        print(f"  [ADAPT-VQE] ⚠ Semua restart selesai. "
              f"Menggunakan hasil terbaik: E = {best_e:.8f} Ha")
        self.timing_ = best_timing
        self.circuit = best_circuit
        return best_e, best_circuit, best_energies

    def _run_single_pass(self, estimator, optimizer_fn, perturb_scale: float,
                         rng) -> tuple:
        """
        Satu pass penuh ADAPT-VQE. Dipanggil ulang oleh run() jika hasil
        melanggar variational principle.

        perturb_scale : skala noise random untuk parameter awal tiap operator
                        baru (0.0 = inisialisasi standar/nol)
        """
        energies         = []
        iter_times        = []
        opt_times         = []
        grad_times        = []
        param_vals        = []
        e_min             = float("inf")
        self._used_indices = set()
        self.selected_ops  = []
        self.params        = []

        plateau_count = getattr(config, "ADAPT_PLATEAU_N",   5)
        energy_tol    = getattr(config, "ADAPT_PLATEAU_TOL", 1e-8)
        stagnant      = 0
        t_run_start   = time.time()

        for it in range(self.max_iter):
            t_iter_start = time.time()

            # ── Statevector & gradien ─────────────────────────────────────────
            psi = self._get_statevector(param_vals)
            t_grad_start = time.time()
            grads    = self._compute_gradients(psi)
            t_grad   = time.time() - t_grad_start
            grad_times.append(t_grad)

            max_grad = float(np.max(grads))
            print(f"  [ADAPT iter {it+1:3d}] max|grad|={max_grad:.6f} "
                  f"(grad {t_grad:.1f}s)", end="", flush=True)

            if max_grad < self.grad_tol:
                print(f" → Konvergen grad (< {self.grad_tol})")
                iter_times.append(time.time() - t_iter_start)
                break

            # ── Tambah operator ───────────────────────────────────────────────
            idx_best = int(np.argmax(grads))
            self._used_indices.add(idx_best)
            self.selected_ops.append(self.pool[idx_best])
            self.params.append(Parameter(f"θ_adapt_{len(self.params)}"))

            # Inisialisasi parameter baru: nol, atau diperturbasi jika retry
            x0_new = 0.0 if perturb_scale == 0.0 else rng.normal(0, perturb_scale)
            param_vals.append(x0_new)

            # ── Optimasi ──────────────────────────────────────────────────────
            ansatz_qc = self._build_circuit_parametric()
            _params   = list(self.params)

            def cost(x, qc=ansatz_qc, ps=_params):
                bound = qc.assign_parameters(dict(zip(ps, x)))
                job   = estimator.run([(bound, self.hamiltonian)])
                return float(job.result()[0].data.evs.real)

            t_opt_start = time.time()
            opt_result  = optimizer_fn(cost, np.array(param_vals))
            t_opt       = time.time() - t_opt_start
            opt_times.append(t_opt)

            param_vals = list(opt_result.x)
            e_new      = float(opt_result.fun)

            # ── Plateau check ─────────────────────────────────────────────────
            delta_e = abs(e_new - e_min) if e_min != float("inf") else abs(e_new)
            stagnant = stagnant + 1 if delta_e < energy_tol else 0
            e_min = e_new
            energies.append(e_min)

            t_iter = time.time() - t_iter_start
            iter_times.append(t_iter)

            print(f" | E={e_min:.8f} Ha  ΔE={delta_e:.2e} "
                  f"| opt {t_opt:.1f}s | iter {t_iter:.1f}s", end="")

            if stagnant >= plateau_count:
                print(f" → Plateau ({plateau_count}×)")
                break
            print()

        t_total = time.time() - t_run_start

        # ── Sirkuit final ─────────────────────────────────────────────────────
        if self.selected_ops:
            final_qc = self._build_circuit_parametric()
            circuit  = final_qc.assign_parameters(
                dict(zip(self.params, param_vals))
            )
        else:
            circuit = self.hf_circuit.copy()

        e_final = energies[-1] if energies else float("nan")
        n_iter  = len(energies)

        if opt_times:
            print(f"\n[ADAPT-VQE] Selesai: {n_iter} iter | {len(self.selected_ops)} op | "
                  f"E={e_final:.8f} Ha | total {t_total:.1f}s "
                  f"(grad avg {np.mean(grad_times):.1f}s, "
                  f"opt avg {np.mean(opt_times):.1f}s)")
        else:
            print(f"\n[ADAPT-VQE] Selesai: 0 iter | E={e_final:.8f} Ha")

        timing = {
            "total_s"      : t_total,
            "iter_times_s" : iter_times,
            "opt_times_s"  : opt_times,
            "grad_times_s" : grad_times,
            "n_iter"       : n_iter,
            "mean_grad_s"  : float(np.mean(grad_times)) if grad_times else 0.0,
            "mean_opt_s"   : float(np.mean(opt_times))  if opt_times  else 0.0,
        }
        return e_final, circuit, energies, timing


# ──────────────────────────────────────────────────────────────────────────────
# Fungsi dispatcher utama
# ──────────────────────────────────────────────────────────────────────────────

def build_ansatz(
    n_qubits: int,
    hamiltonian: SparsePauliOp = None,
    ansatz_type: str = None,
    encoding: str = None,
):
    """
    Dispatcher: kembalikan ansatz sesuai config.ANSATZ_TYPE.

    Untuk UCCSD dan kUpCCGSD: kembalikan QuantumCircuit.
    Untuk ADAPT-VQE: kembalikan objek AdaptVQE.
    """
    if ansatz_type is None:
        ansatz_type = config.ANSATZ_TYPE
    if encoding is None:
        encoding = config.ENCODING

    atype = ansatz_type.upper()

    if atype == "UCCSD":
        return build_uccsd(n_qubits, encoding)

    elif atype in ("KUPCCGSD", "K-UPCCGSD"):
        return build_kupccgsd(n_qubits, config.K_UPCCGSD, encoding)

    elif atype in ("ADAPT-VQE", "ADAPTVQE", "ADAPT"):
        if hamiltonian is None:
            raise ValueError("ADAPT-VQE membutuhkan hamiltonian untuk menghitung gradien.")
        return AdaptVQE(
            hamiltonian=hamiltonian,
            n_qubits=n_qubits,
            encoding=encoding,
        )

    else:
        raise ValueError(
            f"Ansatz '{ansatz_type}' tidak dikenal. Pilih: UCCSD | kUpCCGSD | ADAPT-VQE"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Test cepat
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    n_q = 2 * config.N_ACTIVE_ORBITALS
    print(f"Ansatz type: {config.ANSATZ_TYPE}")
    if config.ANSATZ_TYPE != "ADAPT-VQE":
        ans = build_ansatz(n_q)
        print(ans)
    else:
        print("[ADAPT-VQE] Perlu Hamiltonian — jalankan via vqe_runner.py")