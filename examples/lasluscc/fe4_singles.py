"""Four-fragment Fe-S single-excitation LAS-LUSCC benchmark.

Edit reference_dir to a saved Fe4 reference directory containing reference.chk
and matching integrals.npz (for example, references/s1 from the Fe4 handoff).
The singles use fixed orbital order, not energy-lowering rankings.
"""
from functools import partial
from pathlib import Path
from unittest import mock

import numpy as np
from pyscf import lib, scf
from pyscf.lib import chkfile

from mrh.exploratory.luscc import LUSCC
from mrh.my_pyscf.lassi import LASSI
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF


reference_dir = Path("references/s1")
n_singles = 10
max_memory = 24000  # MB
lib.num_threads(4)
assert 0 <= n_singles <= 300

# Four Fe-centered fragments: 22 electrons in 20 active orbitals.
checkpoint = str(reference_dir / "reference.chk")
mol = chkfile.load_mol(checkpoint)
mol.verbose = 0
mol.max_memory = max_memory
las = LASSCF(scf.ROHF(mol).density_fit(), (5, 5, 5, 5), (6, 5, 6, 5),
             spin_sub=(5, 6, 5, 6))
ref = LASSI(las)
with mock.patch.object(las, "state_average",
                       partial(las.state_average, assert_no_dupes=False)):
    ref.load_chk(checkpoint)
target_spin = int(round((np.sqrt(1 + 4 * ref.s2[0]) - 1) / 2))
np.testing.assert_allclose(ref.s2[0], target_spin * (target_spin + 1), atol=1e-6, rtol=0)
assert tuple(ref.ncas_sub) == (5, 5, 5, 5)

# Interfragment singles with adjacent alpha/beta partners.
# LUSCC automatically includes each generator's reverse excitation.
pairs = [(a + 20 * s, i + 20 * s)
         for i in range(20) for a in range(i + 1, 20)
         if a // 5 != i // 5 for s in range(2)][:n_singles]
a_idxs = [[a] for a, i in pairs]
i_idxs = [[i] for a, i in pairs]
solver = LUSCC(ref, a_idxs, i_idxs, top_m=110, target_spin=target_spin,
               operator_backend="cached")  # Use "original" for comparison.
solver.max_memory = max_memory

# Reuse the saved Hamiltonian rather than regenerating integrals.
with np.load(reference_dir / "integrals.npz") as saved:
    integrals = tuple(saved[k] for k in ("h0", "h1", "h2"))
with mock.patch.object(solver, "ham_2q", return_value=integrals):
    energy, _, s2, residual = solver.kernel()
assert solver.converged, "Projected Davidson did not converge"
np.testing.assert_allclose(s2, target_spin * (target_spin + 1), atol=1e-6, rtol=0)
assert np.max(residual**2) < 1e-7, "Spin residual exceeds tolerance"

print(f"Reference energy: {ref.e_roots[0]:.12f} Eh")
print(f"LAS-LUSCC energy: {energy[0]:.12f} Eh")
print(f"Selected singles: {len(pairs)}; target spin: {target_spin}")
print(f"<S^2>: {s2[0]:.10f}; spin residual norm: {residual[0]:.3e}")
for stage, seconds in solver.timings.items():
    print(f"{stage:24s} {seconds:10.3f} s")
