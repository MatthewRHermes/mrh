"""LAS-LUSCC for butadiene.

This small example compares a localized active-space reference with LASSIS
and a LAS-LUSCC expansion selected from the largest energy gradients.
"""

import numpy as np
from pyscf import gto, mcscf, scf

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
C  1.855098  0.114866  0.000000
C -1.855098 -0.114866  0.000000
C  0.643269 -0.423208  0.000000
C -0.643269  0.423208  0.000000
H  2.022642  1.200276  0.000000
H -2.022642 -1.200276  0.000000
H  2.772605 -0.488763  0.000000
H -2.772605  0.488763  0.000000
H  0.475726 -1.508617  0.000000
H -0.475726  1.508617  0.000000
    """,
    basis="6-31g",
    verbose=4,
)
mf = scf.RHF(mol).run()

# The (4e,4o) active space is divided between the two terminal C=C fragments.
ncas_sub = (2, 2)
nelecas_sub = (2, 2)
las = LASSCF(mf, ncas_sub, nelecas_sub, spin_sub=(1, 1))
mo_guess = las.localize_init_guess(((0, 2), (3, 1)), mf.mo_coeff)
las.kernel(mo_guess)

# Full CASCI and LASSIS provide useful references for this small system.
cas = mcscf.CASCI(mf, sum(ncas_sub), sum(nelecas_sub)).run()
e_lassis, _ = lassi.LASSIS(las).kernel()

# Rank the interfragment singles and doubles by their LAS energy gradients.
gradients, _, a_idxs, i_idxs = get_grad_exact(las)
a_selected, i_selected = select_largest(
    a_idxs, i_idxs, gradients, fraction=0.10
)
e_luscc, _ = LUSCC(las, a_selected, i_selected).kernel()

print(f"CASCI energy:     {cas.e_tot:.12f}")
print(f"LASSCF energy:    {las.e_tot:.12f}")
print(f"LASSIS energy:    {e_lassis[0]:.12f}")
print(f"LAS-LUSCC energy: {e_luscc[0]:.12f}")
print(f"Selected {len(a_selected)} of {len(a_idxs)} excitations")
