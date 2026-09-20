# QuMat-Shield — VQE prototype (Team QForge)

Quantum simulation of material fragments with Qiskit: VQE ground-state energy,
benchmarked against exact diagonalization, plus a bond-stability curve.

## Run (Google Colab or Linux/WSL)
```
pip install -r requirements.txt
python qumat_shield_vqe.py --molecule h2  --mode single   # validation (4 qubits, Jordan-Wigner)
python qumat_shield_vqe.py --molecule h2  --mode scan     # energy vs bond length
python qumat_shield_vqe.py --molecule lih --mode scan     # metal-hydride fragment, 6 qubits
```
Results (plots, CSV, JSON) go to `./results`.

## Sanity check
H2, STO-3G, 0.735 Å: exact total energy ≈ -1.1373 Hartree.

## Scope
Proof-of-concept on small fragments. Real defence-material lattices need far more qubits;
scaling plan = active-space reduction, error mitigation, real hardware (see slide 8/10).
