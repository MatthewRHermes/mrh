from mrh.my_pyscf.pbc.fci import direct_com_real
from mrh.my_pyscf.pbc.fci import rdm_helper


def solver(cell, singlet, symm=None):
    # Will add the singlet and symm options later.
    return direct_com_real.FCISolver(cell)