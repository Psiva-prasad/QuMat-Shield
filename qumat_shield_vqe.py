"""
QuMat-Shield  |  Team QForge (Siva Prasad)
Qiskit VQE prototype: ground-state energy + bond-stability curve of material fragments.

Pipeline (matches slide 7 of the presentation)
  1  Candidate fragment (H2 or LiH)          -> PySCF via Qiskit Nature driver
  2  Qubit Hamiltonian (Jordan-Wigner)       -> Qiskit Nature mapper
  3  VQE: UCCSD ansatz + Estimator           -> QUANTUM step (Qiskit)
  4  Classical optimizer (SciPy)             -> updates theta, loop until converged
  5  Ground-state energy                     -> E_VQE
  6  Classical benchmark (exact diagonalization) -> E_exact, error in Hartree
  7  Materials insight                       -> equilibrium bond length, well depth

Usage
  python qumat_shield_vqe.py --molecule h2  --mode single            # quick validation
  python qumat_shield_vqe.py --molecule h2  --mode scan              # bond-stability curve
  python qumat_shield_vqe.py --molecule lih --mode scan --optimizer SLSQP

Outputs (folder ./results): *.png plots, *.csv data, *.json summary.
"""
import argparse
import csv
import json
import os
import time

import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CHEMICAL_ACCURACY = 1.6e-3  # Hartree (~1 kcal/mol)
HARTREE_TO_EV = 27.211386


# ----------------------------------------------------------------------
# Step 1: candidate fragment -> electronic-structure problem
# ----------------------------------------------------------------------
def build_problem(molecule: str, distance: float):
    from qiskit_nature.units import DistanceUnit
    from qiskit_nature.second_q.drivers import PySCFDriver
    from qiskit_nature.second_q.transformers import ActiveSpaceTransformer

    if molecule == "h2":
        atom, transformer = f"H 0 0 0; H 0 0 {distance}", None
    elif molecule == "lih":  # metal-hydride fragment, active space (2e, 3 orbitals)
        atom = f"Li 0 0 0; H 0 0 {distance}"
        transformer = ActiveSpaceTransformer(num_electrons=2, num_spatial_orbitals=3)
    else:
        raise ValueError("molecule must be 'h2' or 'lih'")

    driver = PySCFDriver(atom=atom, basis="sto3g", charge=0, spin=0,
                         unit=DistanceUnit.ANGSTROM)
    problem = driver.run()
    if transformer is not None:
        problem = transformer.transform(problem)
    return problem


# ----------------------------------------------------------------------
# Steps 2-5: VQE (quantum energy evaluation + classical optimizer loop)
# ----------------------------------------------------------------------
def run_vqe(problem, mapper, optimizer="L-BFGS-B", maxiter=500, tol=1e-12, x0=None):
    from qiskit.primitives import StatevectorEstimator
    from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD

    qubit_op = mapper.map(problem.hamiltonian.second_q_op())          # step 2
    shift = float(sum(problem.hamiltonian.constants.values()))        # nuclear repulsion etc.

    hf_state = HartreeFock(problem.num_spatial_orbitals, problem.num_particles, mapper)
    ansatz = UCCSD(problem.num_spatial_orbitals, problem.num_particles, mapper,
                   initial_state=hf_state)                            # step 3 (ansatz)
    estimator = StatevectorEstimator()                                # exact expectation values

    history = []

    def cost(theta):
        pub = (ansatz, qubit_op, np.asarray(theta, dtype=float))
        energy = float(estimator.run([pub]).result()[0].data.evs)     # QUANTUM step
        history.append(energy + shift)
        return energy

    if x0 is None:
        x0 = np.zeros(ansatz.num_parameters)                          # starts at Hartree-Fock
    t0 = time.time()
    res = minimize(cost, x0, method=optimizer, tol=tol, options={"maxiter": maxiter})  # step 4
    return {
        "energy": float(res.fun) + shift,                             # step 5
        "hf_energy": history[0],
        "theta": res.x,
        "n_qubits": int(ansatz.num_qubits),
        "n_params": int(ansatz.num_parameters),
        "n_iterations": int(getattr(res, "nit", len(history))),
        "n_energy_evals": len(history),
        "history": history,
        "seconds": time.time() - t0,
    }


# ----------------------------------------------------------------------
# Step 6: classical benchmark (exact diagonalization)
# ----------------------------------------------------------------------
def exact_energy(problem, mapper):
    from qiskit_algorithms import NumPyMinimumEigensolver
    from qiskit_nature.second_q.algorithms import GroundStateEigensolver

    solver = GroundStateEigensolver(mapper, NumPyMinimumEigensolver())
    return float(np.real(solver.solve(problem).total_energies[0]))


# ----------------------------------------------------------------------
# Step 7: materials insight from an energy-vs-distance curve
# ----------------------------------------------------------------------
def fit_minimum(d, e):
    """Equilibrium bond length and minimum energy via a parabola through the lowest points."""
    d, e = np.asarray(d, float), np.asarray(e, float)
    i = int(np.argmin(e))
    lo = max(i - 1, 0)
    hi = min(lo + 3, len(d))
    lo = max(hi - 3, 0)
    c = np.polyfit(d[lo:hi], e[lo:hi], 2)
    if c[0] <= 0:
        return float(d[i]), float(e[i])
    d_eq = -c[1] / (2 * c[0])
    return float(d_eq), float(np.polyval(c, d_eq))


def summarize(r, e_exact):
    err = abs(r["energy"] - e_exact)
    return {
        "E_vqe_Ha": r["energy"], "E_exact_Ha": e_exact, "E_hf_Ha": r["hf_energy"],
        "abs_error_Ha": err, "abs_error_mHa": err * 1e3,
        "within_chemical_accuracy": bool(err < CHEMICAL_ACCURACY),
        "qubits": r["n_qubits"], "parameters": r["n_params"],
        "optimizer_iterations": r["n_iterations"], "energy_evaluations": r["n_energy_evals"],
        "runtime_s": round(r["seconds"], 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecule", choices=["h2", "lih"], default="h2")
    ap.add_argument("--mode", choices=["single", "scan"], default="single")
    ap.add_argument("--distance", type=float, default=None, help="bond length in Angstrom (single mode)")
    ap.add_argument("--optimizer", default="L-BFGS-B", help="any scipy.optimize.minimize method")
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    from qiskit_nature.second_q.mappers import JordanWignerMapper
    mapper = JordanWignerMapper()

    default_d = {"h2": 0.735, "lih": 1.6}[a.molecule]

    # ---------------- single-point validation ----------------
    if a.mode == "single":
        d = a.distance or default_d
        problem = build_problem(a.molecule, d)
        r = run_vqe(problem, mapper, optimizer=a.optimizer)
        s = summarize(r, exact_energy(problem, mapper))
        s.update({"molecule": a.molecule, "distance_A": d, "optimizer": a.optimizer})
        print(json.dumps(s, indent=2))
        json.dump(s, open(f"{a.outdir}/{a.molecule}_single.json", "w"), indent=2)

        err = np.abs(np.array(r["history"]) - s["E_exact_Ha"]) + 1e-16
        plt.figure(figsize=(6, 4))
        plt.semilogy(err, color="#F237A6", lw=2)
        plt.axhline(CHEMICAL_ACCURACY, ls="--", color="#121216", label="chemical accuracy (1.6 mHa)")
        plt.xlabel("Energy evaluation"); plt.ylabel("|E_VQE - E_exact|  (Hartree)")
        plt.title(f"VQE convergence - {a.molecule.upper()} @ {d} Å")
        plt.legend(); plt.tight_layout()
        plt.savefig(f"{a.outdir}/{a.molecule}_convergence.png", dpi=200)
        return

    # ---------------- bond-stability scan ----------------
    grid = {"h2": np.arange(0.5, 2.55, 0.1), "lih": np.arange(1.0, 3.01, 0.25)}[a.molecule]
    rows, theta = [], None
    for d in grid:
        problem = build_problem(a.molecule, float(d))
        r = run_vqe(problem, mapper, optimizer=a.optimizer, x0=theta)   # warm start from previous point
        theta = r["theta"]
        e_ex = exact_energy(problem, mapper)
        rows.append((float(d), r["energy"], e_ex, abs(r["energy"] - e_ex), r["n_energy_evals"]))
        print(f"d={d:.2f} Å  E_VQE={r['energy']:.8f}  E_exact={e_ex:.8f}  err={abs(r['energy']-e_ex):.2e} Ha")

    d_arr, e_vqe, e_ex = (np.array([r[i] for r in rows]) for i in (0, 1, 2))
    d_eq, e_min = fit_minimum(d_arr, e_vqe)
    depth = float(e_vqe[-1] - e_min)  # relative to the largest separation in the scan
    insight = {
        "molecule": a.molecule, "equilibrium_bond_length_A": round(d_eq, 3),
        "E_min_Ha": e_min, "well_depth_vs_dmax_eV": round(depth * HARTREE_TO_EV, 3),
        "max_abs_error_mHa": float(np.max([r[3] for r in rows]) * 1e3),
    }
    print(json.dumps(insight, indent=2))
    json.dump(insight, open(f"{a.outdir}/{a.molecule}_insight.json", "w"), indent=2)
    with open(f"{a.outdir}/{a.molecule}_scan.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["distance_A", "E_vqe_Ha", "E_exact_Ha", "abs_error_Ha", "energy_evals"]); w.writerows(rows)

    plt.figure(figsize=(6.5, 4.2))
    plt.plot(d_arr, e_ex, color="#121216", lw=2, label="Exact diagonalization")
    plt.plot(d_arr, e_vqe, "o", color="#F237A6", ms=6, label="VQE (Qiskit)")
    plt.axvline(d_eq, ls=":", color="#C4177F", label=f"equilibrium ≈ {d_eq:.2f} Å")
    plt.xlabel("Bond length (Å)"); plt.ylabel("Total energy (Hartree)")
    plt.title(f"Bond-stability curve - {a.molecule.upper()} (STO-3G)")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{a.outdir}/{a.molecule}_bond_curve.png", dpi=200)


if __name__ == "__main__":
    main()
