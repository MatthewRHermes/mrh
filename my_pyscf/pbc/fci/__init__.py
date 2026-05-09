from mrh.my_pyscf.pbc.fci import direct_spin1_cplx
from mrh.my_pyscf.pbc.fci import csf_cplx
from mrh.my_pyscf.pbc.fci import spin_op
from mrh.my_pyscf.pbc.fci import addons


# Author: Bhavnesh Jangid

# TODO: 
# 1. Add the option of direct_spin0 during the initialization.
# 2. Add the point group symmetry option.

try:
    from mrh.my_pyscf.pbc.fci import dmrg_cplx_helper
    DMRGCICPLX = dmrg_cplx_helper.DMRGCICPLX
except ImportError:
        raise ImportError("DMRGCI with complex integrals is not available. " \
        "Please install the DMRGCI module from GitHub.")

def solver(cell, singlet, symm=None):
    # Will add the singlet and symm options later.
    if symm is not None and symm is not False:
        raise NotImplementedError("Symmetry is not implemented for FCI in PBC yet.")
    return direct_spin1_cplx.FCISolver(cell)

def csf_solver(cell, smult, symm=None):
    if symm is not None and symm is not False:
        raise NotImplementedError("Symmetry is not implemented for CSF-FCI in PBC yet.")
    if smult == 1:
        return csf_cplx.FCISolverSpin0(cell, smult)
    else:
        return csf_cplx.FCISolver(cell, smult)