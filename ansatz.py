"""
ansatz.py
=========
Build ansatz VQE: UCCSD, kUpCCGSD, dan ADAPT-VQE.
Semua output berupa QuantumCircuit Qiskit yang siap digunakan dalam VQE loop.

[VERSION] 2026-06-30-v4-restart-sanity-check
Fitur: ADAPT-VQE dengan variational principle sanity check + auto-restart
       jika E_VQE > E_HF (indikasi local minimum / optimizer gagal).
"""

from multiprocessing import pool

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
#from vqe_hybrid.diagnose_h2_pool import H_mat


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
        [FIX MENYELURUH] Ganti hand-rolled Pauli algebra (_single_exc/_double_exc)
        dengan FermionicOp + mapper yang SAMA dengan Hamiltonian. Ini menghindari
        3 bug sebelumnya sekaligus:
            (a) mismatch ordering interleaved vs block,
            (b) salah tanda koefisien Pauli di formula double-excitation manual,
            (c) enumerasi creation/annihilation pair yang naif & pecah Z-string
                begitu quadruple index tidak sorted monoton.
        Operator pool sekarang dibangun G = i*(T - T_dagger) via FermionicOp,
        lalu di-map dengan mapper yang identik dengan hamiltonian.py, sehingga
        ordering & tanda otomatis konsisten tanpa hand-roll Pauli string lagi.
    """
        from itertools import combinations
        from qiskit.quantum_info import SparsePauliOp
        from qiskit_nature.second_q.operators import FermionicOp
        import numpy as np

        n_so = 2 * config.N_ACTIVE_ORBITALS
        half = n_so // 2   # block ordering: alpha = [0,half), beta = [half,n_so)
        mapper = _get_mapper(self.encoding)   # HARUS mapper yg sama dgn Hamiltonian
        pool = []
        n_rejected_spin = 0

        def _generator_op(fop_dict):
            T = FermionicOp(fop_dict, num_spin_orbitals=n_so)
            G = T - T.adjoint()
            qubit_G = mapper.map(G).simplify(atol=1e-12)
            if len(qubit_G) == 0:
                return None
            op = (1j * qubit_G).simplify(atol=1e-12)
            return SparsePauliOp(op.paulis, coeffs=np.real(op.coeffs))

        # ── Singles: hanya pasangan spin-orbital dgn spin SAMA (block conv.) ────
        for p, q in combinations(range(n_so), 2):
            if (p < half) != (q < half):
                n_rejected_spin += 1
                continue
            op = _generator_op({f"+_{p} -_{q}": 1.0})
            if op is not None:
                pool.append(op)

        # ── Doubles: creation-pair x annihilation-pair independen, filter Sz ────
        seen_ops = set()
        pairs = list(combinations(range(n_so), 2))
        for (p, q) in pairs:
            for (r, s) in pairs:
                if len({p, q, r, s}) < 4:
                    continue
                key = tuple(sorted([tuple(sorted([p, q])), tuple(sorted([r, s]))]))
                if key in seen_ops:
                    continue
                seen_ops.add(key)

                sz_pq = int(p < half) + int(q < half)
                sz_rs = int(r < half) + int(s < half)
                if sz_pq != sz_rs:
                    n_rejected_spin += 1
                    continue

                op = _generator_op({f"+_{p} +_{q} -_{s} -_{r}": 1.0})
                if op is not None:
                    pool.append(op)

        self.pool = pool
        print(f"[ADAPT-VQE] Pool: {len(pool)} operator spin-conserving "
              f"({n_rejected_spin} ditolak) (n_so={n_so}, singles+doubles) "
              f"[via FermionicOp+mapper, konsisten dgn Hamiltonian]")

    # ── Gradien via parameter shift pada statevector ──────────────────────────
    def _compute_gradients(self, statevector: np.ndarray) -> np.ndarray:
        """
        Hitung |⟨ψ|[H, A_k]|ψ⟩| untuk setiap A_k dalam pool yang belum dipakai.

        [FIX] Operator pool (dari _generator_op) adalah HERMITIAN (koefisien
        real), BUKAN anti-Hermitian. Untuk H dan A_k sama-sama Hermitian,
        komutator [H, A_k] selalu ANTI-Hermitian, sehingga ⟨ψ|[H,A_k]|ψ⟩
        murni IMAJINER. Ambil bagian .imag, bukan .real (yang selalu ~0).
        """
        from qiskit.quantum_info import Statevector

        psi   = statevector
        H_mat = self.hamiltonian.to_matrix(sparse=True)
        Hpsi  = H_mat.dot(psi)

        grads = np.zeros(len(self.pool))
        for k, A_k in enumerate(self.pool):
            if k in self._used_indices:
                grads[k] = 0.0
                continue
            try:
                A_mat = A_k.to_matrix(sparse=True)
                Apsi  = A_mat.dot(psi)
                comm_val = np.vdot(psi, H_mat.dot(Apsi)) - np.vdot(psi, A_mat.dot(Hpsi))
                grads[k] = abs(comm_val.imag)   # <-- diganti dari .real
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
                # Cek apakah ini karena HF state sudah = ground state
                from qiskit.quantum_info import Statevector
                e_hf_check = float(np.real(
                    Statevector(self.hf_circuit).expectation_value(self.hamiltonian)
                ))
                e_gs_check = float(np.linalg.eigvalsh(
                    self.hamiltonian.to_matrix()
                )[0].real)
                if abs(e_hf_check - e_gs_check) < 1e-6:
                    print(f" → HF = ground state (E_HF ≈ E_FCI), tidak perlu ADAPT")
                    energies.append(e_hf_check)
                    e_min = e_hf_check
                else:
                    print(f" → Konvergen grad (< {self.grad_tol})")
                    print(f"  [WARN] Semua gradien nol tapi E_HF={e_hf_check:.6f} ≠ E_GS={e_gs_check:.6f}")
                    print(f"  [WARN] Kemungkinan bug pool/gradien. Cek konvensi operator.")
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

        # Jika tidak ada iterasi (0 operator), gunakan energi HF sebagai fallback
        if np.isnan(e_final) and not self.selected_ops:
            from qiskit.quantum_info import Statevector
            e_final = float(np.real(
                Statevector(self.hf_circuit).expectation_value(self.hamiltonian)
            ))
            energies = [e_final]
            print(f"  [INFO] 0 operator ADAPT → menggunakan E_HF = {e_final:.8f} Ha")
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