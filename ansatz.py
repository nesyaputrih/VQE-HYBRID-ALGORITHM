"""
ansatz.py
=========
Build ansatz VQE: UCCSD, kUpCCGSD, dan ADAPT-VQE.
Semua output berupa QuantumCircuit Qiskit yang siap digunakan dalam VQE loop.
"""

import numpy as np
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
        Build pool operator ADAPT dari qubit excitation operators.

        Setiap elemen pool adalah operator anti-Hermitian Hermitian:
            G_pq   = i(|q><p| - |p><q|)   — single excitation
            G_pqrs = i(|rs><pq| - |pq><rs|) — double excitation

        Direpresentasikan sebagai SparsePauliOp real (koef real)
        yang bisa langsung dipakai PauliEvolutionGate.

        Referensi: Grimsley et al., Nat. Commun. 10, 3007 (2019)
        """
        from qiskit.quantum_info import SparsePauliOp as SPS
        from itertools import combinations

        n_so   = 2 * config.N_ACTIVE_ORBITALS   # jumlah spin-orbital
        n_elec = config.N_ACTIVE_ELECTRONS
        # Occupied: spin-orb 0..n_elec-1 (JW ordering interleaved: α=2i, β=2i+1)
        # Virtual : spin-orb n_elec..n_so-1
        occ  = list(range(n_elec))
        virt = list(range(n_elec, n_so))

        pool = []

        # ── Helper: single qubit excitation a†_p a_q - a†_q a_p (anti-Herm) ──
        def _single_exc(p, q, n):
            """
            a†_p a_q - h.c. dalam JW:
            = (i/2)(X_p Z_{p-1}...Z_{q+1} Y_q - Y_p Z_{p-1}...Z_{q+1} X_q)
            Kembalikan sebagai SparsePauliOp nyata dengan koef 1.
            """
            paulis = []
            coeffs = []
            # Term 1: X_p (Z string) Y_q
            label1 = ["I"] * n
            label1[p] = "X"
            label1[q] = "Y"
            for k in range(min(p, q) + 1, max(p, q)):
                label1[k] = "Z"
            paulis.append("".join(reversed(label1)))
            coeffs.append(0.5)
            # Term 2: Y_p (Z string) X_q
            label2 = ["I"] * n
            label2[p] = "Y"
            label2[q] = "X"
            for k in range(min(p, q) + 1, max(p, q)):
                label2[k] = "Z"
            paulis.append("".join(reversed(label2)))
            coeffs.append(-0.5)
            return SPS(paulis, coeffs=coeffs).simplify(atol=1e-12)

        def _double_exc(p, q, r, s, n):
            """
            a†_p a†_q a_r a_s - h.c. (anti-Hermitian, qubit excitation operator).
            Dekomposisi JW menghasilkan 8 suku Pauli dengan koef real.
            Ref: Arrazola et al., Quantum 6, 742 (2022).
            """
            i, j = sorted([p, q])
            k, l = sorted([r, s])

            # 8 kombinasi XY untuk 4 indeks (i,j,k,l), koef ±1/8
            pauli_patterns = [
                (("X", "Y", "X", "X"),  0.125),
                (("X", "X", "X", "Y"), -0.125),
                (("Y", "X", "X", "X"), -0.125),
                (("X", "X", "Y", "X"),  0.125),
                (("Y", "Y", "Y", "X"),  0.125),
                (("Y", "X", "Y", "Y"), -0.125),
                (("X", "Y", "Y", "Y"), -0.125),
                (("Y", "Y", "X", "Y"),  0.125),
            ]

            paulis = []
            coeffs = []
            for (xi, xj, xk, xl), c in pauli_patterns:
                lbl = ["I"] * n
                lbl[i] = xi
                lbl[j] = xj
                lbl[k] = xk
                lbl[l] = xl
                # Z string di antara setiap pasang indeks yang berurutan
                for m in range(i + 1, j):
                    lbl[m] = "Z"
                for m in range(j + 1, k):
                    lbl[m] = "Z"
                for m in range(k + 1, l):
                    lbl[m] = "Z"
                paulis.append("".join(reversed(lbl)))
                coeffs.append(c)

            return SPS(paulis, coeffs=coeffs).simplify(atol=1e-12)

        # ── Single excitations ────────────────────────────────────────────────
        for i in occ:
            for a in virt:
                op = _single_exc(a, i, n_so)
                if len(op) > 0:
                    pool.append(op)

        # ── Double excitations ────────────────────────────────────────────────
        for i, j in combinations(occ, 2):
            for a, b in combinations(virt, 2):
                op = _double_exc(a, b, i, j, n_so)
                if len(op) > 0:
                    pool.append(op)

        self.pool = pool
        print(f"[ADAPT-VQE] Pool: {len(pool)} operator "
              f"({len(occ)} occ, {len(virt)} virt) [{self.pool_type}]")

    # ── Gradien via parameter shift ───────────────────────────────────────────
    def _compute_gradients(self, estimator, current_circuit: QuantumCircuit,
                           param_vals: list) -> np.ndarray:
        """
        Hitung |∂E/∂θ|_{θ=0} untuk exp(θ·iA_k) via parameter shift rule.

        ∂E/∂θ|_{θ=0} = [E(+π/2) - E(-π/2)] / 2
        ekuivalen dengan ⟨ψ|[H, iA_k]|ψ⟩, menghindari operator kosong.
        """
        from qiskit.circuit.library import PauliEvolutionGate
        from qiskit.synthesis import LieTrotter

        # Bind nilai parameter saat ini ke sirkuit dasar
        if self.params and param_vals:
            bound_base = current_circuit.assign_parameters(
                dict(zip(self.params, param_vals))
            )
        else:
            bound_base = current_circuit.copy()

        theta = Parameter("_θ_grad")
        grads = []

        for A_k in self.pool:
            try:
                qc_test = bound_base.copy()
                evo = PauliEvolutionGate(A_k, time=theta,
                                         synthesis=LieTrotter(reps=1))
                qc_test.append(evo, range(self.n_qubits))
                qc_test = qc_test.decompose(reps=2)

                qc_p = qc_test.assign_parameters({theta:  np.pi / 2})
                qc_m = qc_test.assign_parameters({theta: -np.pi / 2})

                job = estimator.run([
                    (qc_p, self.hamiltonian),
                    (qc_m, self.hamiltonian),
                ])
                res  = job.result()
                e_p  = float(res[0].data.evs.real)
                e_m  = float(res[1].data.evs.real)
                grads.append(abs((e_p - e_m) / 2.0))
            except Exception:
                grads.append(0.0)

        return np.array(grads)

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
    def run(self, estimator, optimizer_fn) -> Tuple[float, QuantumCircuit, list]:
        """
        Jalankan loop ADAPT-VQE.

        Kriteria konvergensi (salah satu terpenuhi):
          1. max|grad| < grad_tol
          2. Energi tidak berubah > energy_tol selama plateau_count iterasi berturut-turut

        Returns
        -------
        e_min    : energi VQE minimum
        circuit  : QuantumCircuit ansatz final
        energies : list energi per iterasi ADAPT
        """
        energies      = []
        param_vals    = []
        e_min         = float("inf")
        plateau_count = getattr(config, "ADAPT_PLATEAU_N",   5)
        energy_tol    = getattr(config, "ADAPT_PLATEAU_TOL", 1e-8)
        stagnant      = 0

        for it in range(self.max_iter):
            # ── Sirkuit saat ini ──────────────────────────────────────────────
            current_qc = self._build_circuit_parametric() if self.selected_ops \
                         else self.hf_circuit.copy()

            # ── Hitung gradien via parameter shift ────────────────────────────
            grads    = self._compute_gradients(estimator, current_qc, param_vals)
            max_grad = float(np.max(grads))
            print(f"  [ADAPT iter {it+1:3d}] max|grad| = {max_grad:.6f}", end="")

            if max_grad < self.grad_tol:
                if len(self.selected_ops) == 0:
                    print(f" → grad kecil, ansatz kosong → paksa tambah 1 operator")
                    # jangan break, biarkan loop lanjut ke bagian penambahan operator
                else:
                    print(f" → Konvergen grad (< {self.grad_tol})")
                break

            # ── Tambahkan operator dengan gradien terbesar ────────────────────
            idx_best = int(np.argmax(grads))
            self.selected_ops.append(self.pool[idx_best])
            self.params.append(Parameter(f"θ_adapt_{len(self.params)}"))
            param_vals.append(0.0)

            # ── Optimalkan semua parameter ────────────────────────────────────
            ansatz_qc = self._build_circuit_parametric()
            _params   = list(self.params)

            def cost(x, qc=ansatz_qc, ps=_params):
                bound = qc.assign_parameters(dict(zip(ps, x)))
                job   = estimator.run([(bound, self.hamiltonian)])
                return float(job.result()[0].data.evs.real)

            opt_result = optimizer_fn(cost, np.array(param_vals))
            param_vals = list(opt_result.x)
            e_new      = float(opt_result.fun)

            # ── Cek plateau energi ────────────────────────────────────────────
            if abs(e_new - e_min) < energy_tol:
                stagnant += 1
            else:
                stagnant = 0

            e_min = e_new
            energies.append(e_min)
            print(f" | E = {e_min:.8f} Ha", end="")

            if stagnant >= plateau_count:
                print(f" → Plateau energi ({plateau_count}× tidak berubah > {energy_tol:.0e} Ha)")
                break
            else:
                print()  # newline normal

        # Bind parameter final
        if self.selected_ops:
            final_qc = self._build_circuit_parametric()
            self.circuit = final_qc.assign_parameters(
                dict(zip(self.params, param_vals))
            )
        else:
            self.circuit = self.hf_circuit.copy()

        e_final = energies[-1] if energies else float("nan")
        print(f"[ADAPT-VQE] Selesai: {len(self.selected_ops)} operator, "
              f"E = {e_final:.8f} Ha")
        return e_final, self.circuit, energies


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