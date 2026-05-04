from mrh.my_pyscf.pbc.fci import direct_spin1_cplx
from mrh.my_pyscf.pbc.fci import csf_cplx
from mrh.my_pyscf.pbc.fci import spin_op
from mrh.my_pyscf.pbc.fci import addons
try:
    from pyscf import dmrgscf
    from mrh.my_pyscf.pbc.fci import dmrgci_cplx_helper
    DMRGCIComplex = dmrgci_cplx_helper.DMRGCIComplex
except ImportError:
    pass

def solver(cell, singlet, symm=None):
    # Will add the singlet and symm options later.
    return direct_spin1_cplx.FCISolver(cell)

def csf_solver(cell, smult, symm=None):
    if smult == 1:
        return csf_cplx.FCISolverSpin0(cell, smult)
    else:
        return csf_cplx.FCISolver(cell, smult)