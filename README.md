# VQE-HYBRID-ALGORITHM

Program simulasi kimia kuantum berbasis Variational Quantum Eigensolver (VQE)
untuk molekul H₂, H₂O, dan CH₄. Menggabungkan pipeline klasik (CASCI/CASSCF
via PySCF) dengan simulasi kuantum (Qiskit).

---

## Struktur Folder

```
vqe_hybrid/
├── config.py           # [1] Semua parameter simulasi (EDIT DI SINI)
├── molecule.py         # [2] Geometri & optimasi molekul
├── hamiltonian.py      # [3] Build Hamiltonian qubit (CASCI/CASSCF + JW/PARITY/BK)
├── ansatz.py           # [4] Ansatz UCCSD / kUpCCGSD / ADAPT-VQE
├── initial_state.py    # [5] Inisialisasi HF state & parameter
├── optimizer.py        # [6] Optimizer BFGS, L-BFGS-B, COBYLA, RMSProp, NFT
├── vqe_runner.py       # [7] Loop VQE & scan PES
├── plot_results.py     # [8] Plot PES & error
├── run_hpc.sh          # SLURM script (satu molekul)
├── run_hpc_array.sh    # SLURM array job (H2 + H2O + CH4 paralel)
├── setup_env.sh        # Instalasi environment conda
├── results/            # Output JSON energi VQE (auto-dibuat)
└── plots/              # Output gambar PNG/PDF (auto-dibuat)
```

---

## Cara Penggunaan

### 1. Setup environment (sekali saja)

```bash
chmod +x setup_env.sh
./setup_env.sh
conda activate mycenvn
```

### 2. Konfigurasi parameter

Edit `config.py` untuk memilih:

| Parameter | Opsi |
|-----------|------|
| `MOLECULE` | `"H2"` \| `"H2O"` \| `"CH4"` |
| `ACTIVE_SPACE_METHOD` | `"CASCI"` \| `"CASSCF"` |
| `N_ACTIVE_ELECTRONS` | integer (misal: H2O → 4) |
| `N_ACTIVE_ORBITALS` | integer (misal: H2O → 4) |
| `ENCODING` | `"JW"` \| `"PARITY"` \| `"BK"` |
| `ANSATZ_TYPE` | `"UCCSD"` \| `"kUpCCGSD"` \| `"ADAPT-VQE"` |
| `OPTIMIZER_TYPE` | `"BFGS"` \| `"L-BFGS-B"` \| `"COBYLA"` \| `"RMSProp"` \| `"NFT"` |

### 3. Jalankan lokal

```bash
python vqe_runner.py     # scan PES lengkap
python plot_results.py   # buat plot
```

### 4. Submit ke HPC BRIN

```bash
# Satu molekul (sesuai config.py)
sbatch run_hpc.sh

# Override molekul via argumen
sbatch --export=ALL,MOL=H2,ANSATZ=UCCSD,ENC=JW run_hpc.sh

# Tiga molekul sekaligus (paralel array job)
sbatch run_hpc_array.sh
```

---

## Active Space yang Direkomendasikan

| Molekul | N_elec | N_orbs | N_frozen | Qubit (JW) | Basis |
|---------|--------|--------|----------|------------|-------|
| H₂ | 2 | 2 | 0 | 4 | sto-3g |
| H₂O | 4 | 4 | 1 | 8 | sto-3g |
| CH₄ | 8 | 8 | 1 | 16 | sto-3g |

---

## Output

- **`results/<prefix>_pes.json`** — Energi VQE, FCI, CASSCF, HF di setiap panjang ikatan
- **`plots/<prefix>_pes.png`** — Kurva PES (energi vs r)
- **`plots/<prefix>_error.png`** — Error |E_VQE - E_FCI| vs r + garis chemical accuracy
- **`plots/<prefix>_convergence_req.png`** — Konvergensi optimizer di r ekuilibrium

---

## Dependensi Utama

- `pyscf >= 2.6` — kalkulasi elektronik klasik (RHF, CASCI, CASSCF, FCI)
- `openfermion >= 1.6` — konversi operator fermionik
- `openfermionpyscf >= 0.5` — integrasi PySCF-OpenFermion
- `qiskit >= 1.1` — sirkuit kuantum
- `qiskit-nature >= 0.7` — ansatz kimia kuantum
- `scipy >= 1.13` — optimizer klasik
- `matplotlib >= 3.9` — visualisasi

---

## Referensi

- Peruzzo et al., *Nat. Commun.* **5**, 4213 (2014) — VQE
- Grimsley et al., *Nat. Commun.* **10**, 3007 (2019) — ADAPT-VQE
- Tilly et al., *Phys. Rep.* **986**, 1–128 (2022) — Review VQE
- McClean et al., *New J. Phys.* **18**, 023023 (2016) — VQE theory
- Nakanishi et al., *Phys. Rev. Research* **2**, 043158 (2020) — NFT/RotoSolve
