"""LAS-LUSCC for stilbene at a 90-degree central dihedral.

The active space is split into the two phenyl pi systems and the central
ethylene pi bond.  A small, gradient-selected cluster expansion keeps this
larger demonstration tractable.
"""

import numpy as np
from pyscf import gto, scf

from mrh.exploratory.luscc import LUSCC, get_grad_exact
from mrh.my_pyscf import lassi
from mrh.my_pyscf.mcscf.lasscf_sync_o0 import LASSCF


def select_largest(a_idxs, i_idxs, gradients, fraction):
    """Return the fraction of excitations with the largest |gradient|."""
    nselect = max(1, int(np.ceil(fraction * len(gradients))))
    order = np.argsort(-np.abs(gradients))[:nselect]
    return [a_idxs[i] for i in order], [i_idxs[i] for i in order]


mol = gto.M(
    atom="""
C    0.6125    1.4765    0.3848
C    1.7122    0.6592    0.1075
C    1.6193   -0.3700   -0.8716
C    2.6863   -1.2100   -1.1127
C    3.8573   -1.0936   -0.3748
C    3.9580   -0.1107    0.6137
C    2.9212    0.7529    0.8468
C   -0.6119    1.4761   -0.3851
C   -1.7119    0.6593   -0.1078
C   -1.6198   -0.3689    0.8740
C   -2.6870   -1.2090    1.1136
C   -3.8575   -1.0930    0.3753
C   -3.9574   -0.1113   -0.6152
C   -2.9205    0.7520   -0.8478
H    0.6927    2.1481    1.2377
H    0.6835   -0.5004   -1.3996
H    2.5990   -1.9861   -1.8614
H    4.6835   -1.7822   -0.5455
H    4.8743   -0.0269    1.1961
H    3.0071    1.5194    1.6083
H   -0.6921    2.1457   -1.2397
H   -0.6847   -0.4983    1.3997
H   -2.6003   -1.9830    1.8628
H   -4.6839   -1.7811    0.5458
H   -4.8731   -0.0284   -1.1968
H   -3.0060    1.5174   -1.6112
    """,
    basis="6-31g",
    max_memory=400_000,
    verbose=4,
)
mf = scf.RHF(mol).run()

ncas_sub = (4, 2, 4)
nelecas_sub = (4, 2, 4)
frag_atoms = (
    (1, 2, 3, 4, 5, 6, 15, 16, 17, 18, 19),
    (0, 7, 14, 20),
    (8, 9, 10, 11, 12, 13, 21, 22, 23, 24, 25),
)
las = LASSCF(mf, ncas_sub, nelecas_sub, spin_sub=(1, 1, 1))
mo_guess = las.localize_init_guess(frag_atoms, mf.mo_coeff)
las.kernel(mo_guess)

e_lassis, _ = lassi.LASSIS(las).kernel()
gradients, _, a_idxs, i_idxs = get_grad_exact(las)
a_selected, i_selected = select_largest(
    a_idxs, i_idxs, gradients, fraction=0.01
)
luscc = LUSCC(las, a_selected, i_selected)
e_luscc, _ = luscc.kernel()
s2 = float(np.real(luscc.s2[0]))
spin = (np.sqrt(1.0 + 4.0*s2) - 1.0) / 2.0
multiplicity = 2.0*spin + 1.0

print(f"LASSCF energy:    {las.e_tot:.12f}")
print(f"LASSIS energy:    {e_lassis[0]:.12f}")
print(f"LAS-LUSCC energy: {e_luscc[0]:.12f}")
print(f"LAS-LUSCC <S^2>:  {s2:.12f}")
print(f"LAS-LUSCC 2S+1:   {multiplicity:.12f}")
print(f"Selected {len(a_selected)} of {len(a_idxs)} excitations")
