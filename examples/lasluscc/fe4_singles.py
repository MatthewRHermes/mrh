"""Self-contained four-fragment Fe-S single-excitation LAS-LUSCC benchmark.

Builds ROHF/LASSCF orbitals and a 110-product-state LASSI reference from scratch.
No checkpoint, XYZ, Molden, or integral input files are required. Freshly
optimized orbitals need not reproduce the energies of the saved Fe4 study.
"""
import numpy as np
from pyscf import gto, lib, scf
from pyscf.mcscf import avas

from mrh.exploratory.luscc import LUSCC
from mrh.my_pyscf.lassi import LASSI
from mrh.my_pyscf.lassi.spaces import spin_shuffle, spin_shuffle_ci
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF


spins = range(1, 10)  # Use [1] for one spin, or range(10) to include S=0.
n_singles = 10
max_memory = 24000  # MB; larger excitation spaces require more RAM.
lib.num_threads(4)
assert 0 <= n_singles <= 300

# Chan Fe4S4 geometry (angstrom); the Fe atom indices are 4, 5, 6, 7.
mol = gto.M(
    atom="""
S 0.04 -1.78 -1.29
S -0.04 1.78 -1.29
S 1.78 -0.04 1.29
S -1.78 0.04 1.29
Fe 0.05 -1.37 1.01
Fe -1.38 0.05 -1.00
Fe -0.05 1.38 1.00
Fe 1.37 -0.05 -1.01
S 0.24 3.30 2.14
S -0.24 -3.29 2.14
S -3.29 -0.24 -2.14
S 3.29 0.24 -2.14
C -3.80 -1.84 -1.38
H -3.91 -1.71 -0.29
H -4.76 -2.17 -1.81
H -3.03 -2.60 -1.56
C 3.80 1.83 -1.38
H 3.91 1.71 -0.29
H 4.76 2.16 -1.81
H 3.03 2.59 -1.55
C -1.83 -3.80 1.38
H -2.16 -4.76 1.81
H -2.59 -3.03 1.55
H -1.70 -3.91 0.29
C 1.84 3.80 1.38
H 2.17 4.76 1.81
H 2.60 3.03 1.56
H 1.71 3.91 0.29
    """,
    basis={"C": "cc-pvdz", "H": "cc-pvdz",
           "S": "aug-cc-pvdz", "Fe": "aug-cc-pvdz"},
    unit="Angstrom", charge=-2, spin=18, symmetry=False,
    max_memory=max_memory, verbose=4,
)
mf = scf.ROHF(mol).density_fit().newton()
mf.init_guess = "atom"
mf.kernel()
assert mf.converged, "ROHF did not converge"

# Generate Fe 3d active orbitals instead of loading converged.molden.
# Retain all 18 high-spin singly occupied orbitals in the AVAS guess.
ncas, nelecas, mo_guess = avas.avas(mf, ["Fe 3d"], openshell_option=3)
assert (ncas, nelecas) == (20, 22), (
    f"AVAS selected ({nelecas}e,{ncas}o), expected (22e,20o); inspect the active-space guess")
las = LASSCF(mf, (5, 5, 5, 5), ((5, 1), (5, 0), (1, 5), (0, 5)),
             spin_sub=(5, 6, 5, 6))
mo_guess = las.localize_init_guess(([4], [5], [6], [7]), mo_guess)
las.kernel(mo_guess)
assert las.converged, "LASSCF did not converge"

# Local spins (2, 5/2, 2, 5/2): all 110 products with total Ms=0.
# Rotate the converged fragment CI vectors without reoptimizing their orbitals.
las_m0 = spin_shuffle(las)
las_m0.ci = spin_shuffle_ci(las_m0, las_m0.ci)
las_m0.converged = las.converged
assert las_m0.nroots == 110
ref = LASSI(las_m0).run(nroots_si=110, davidson_only=False)
assert ref.converged, "Reference LASSI did not converge"
assert ref.si.shape[0] == 110

# Fixed interfragment singles, with adjacent alpha/beta partners.
# LUSCC includes each generator's reverse automatically. These are not ranked.
pairs = [(a + 20 * s, i + 20 * s)
         for i in range(20) for a in range(i + 1, 20)
         if a // 5 != i // 5 for s in range(2)][:n_singles]
a_idxs = [[a] for a, i in pairs]
i_idxs = [[i] for a, i in pairs]

for target_spin in spins:
    roots = np.flatnonzero(np.isclose(ref.s2, target_spin * (target_spin + 1),
                                     atol=1e-6, rtol=0))
    assert len(roots), f"No reference root with S={target_spin}"
    state = int(roots[np.argmin(ref.e_roots[roots])])
    solver = LUSCC(ref, a_idxs, i_idxs, state=state, top_m=110,
                   target_spin=target_spin, operator_backend="cached")
    solver.max_memory = max_memory
    energy, _, s2, residual = solver.kernel()
    assert solver.converged, "Projected Davidson did not converge"
    np.testing.assert_allclose(s2, target_spin * (target_spin + 1), atol=1e-6, rtol=0)
    assert np.max(residual**2) < 1e-7, "Spin residual exceeds tolerance"

    print(f"\nS={target_spin}; selected singles: {len(pairs)}")
    print(f"110-state LASSI energy: {ref.e_roots[state]:.12f} Eh")
    print(f"LAS-LUSCC energy:       {energy[0]:.12f} Eh")
    print(f"<S^2>: {s2[0]:.10f}; spin residual norm: {residual[0]:.3e}")
    for stage, seconds in solver.timings.items():
        print(f"{stage:24s} {seconds:10.3f} s")
