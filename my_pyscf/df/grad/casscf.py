import numpy as np
from scipy import linalg
from pyscf import gto
from pyscf import lib
from pyscf import scf
from pyscf import df
from pyscf.ao2mo import _ao2mo
from pyscf.grad.rhf import GradientsBasics
from pyscf.df.grad.rhf import _int3c_wrapper
from pyscf.ao2mo.outcore import balance_partition
from pyscf.ao2mo.incore import _conc_mos

def solve_df_rdm2 (mc_or_mc_grad, mo_coeff=None, ci=None, casdm2=None):
    ''' Solve (P|Q)d_Qij = (P|kl)d_ijkl for d_Qij in the MO basis.

    Args:
        mc_or_mc_grad: DF-MCSCF energy or gradients method object.

    Kwargs:
        mo_coeff: ndarray, tuple, or list containing mo_coefficients.
            If two ndarrays mo_coeff = (mo0, mo1) are provided, mo0 and mo1 are
            assumed to correspond to casdm2's leading and second dimension,
            respectively, regardless of len (ci) or len (casdm2).
            (This will facilitate SA-CASSCF gradients at some point.)
        ci: ndarray, tuple, or list containing CI coefficients in mo_coeff basis.
            Not used if casdm2 is provided.
        casdm2: ndarray, tuple, or list containing rdm2 in mo_coeff basis.
            Computed by mc_or_mc_grad.fcisolver.make_rdm12 (ci,...) if omitted.
        
    Returns:
        dfcasdm2: ndarray or list containing 3-center 2RDM, d_Pqr, where P is
            auxbasis index and q, r are mo_coeff basis indices. '''


    # Initialize casdm2, mo_coeff, and nset
    if mo_coeff is None: mo_coeff = mc_or_mc_grad.mo_coeff
    if ci is None: ci = mc_or_mc_grad.ci
    if casdm2 is None:
        ncas = mc_or_mc_grad.ncas
        nelecas = mc_or_mc_grad.nelecas
        casdm2 = mc_or_mc_grad.fcisolver.make_rdm12 (ci, ncas, nelecas)
    if np.asarrary (mo_coeff).ndim == 4:
        casdm2 = [casdm2]
    nset = len (casdm2)

    # Initialize mol and auxmol
    mol = mc_or_mc_grad.mol
    if isinstance (mc_or_mc_grad, GradientsBasics):
        with_df = mc_or_mc_grad.base.with_df
    else:
        mc = mc_or_mc_grad.with_df
    auxmol = with_df.auxmol
    if auxmol is None:
        auxmol = df.addons.make_auxmol(with_df.mol, with_df.auxbasis)
    nao, naux, nbas, nauxbas = mol.nao, auxmol.nao, mol.nbas, auxmol.nbas
    npair = nao * (nao + 1) // 2

    # Separate mo_coeff
    if isinstance (mo_coeff, np.ndarray) and mo_coeff.ndim == 2:
        mo0 = mo1 = mo_coeff
    else:
        mo0, mo1 = mo_coeff[0], mo_coeff[1]
    nmo0, nmo1 = mo0.shape[-1], mo1.shape[-1]
    mosym, nmo_pair, mo_conc, mo_slice = _conc_mos(mo0, mo1, True)

    # (P|uv) -> (P|ij)
    get_int3c = _int3c_wrapper(mol, auxmol, 'int3c2e', 's2ij')
    int3c = np.zeros ((naux, nmo_pair), dtype=mo0.dtype)
    buf = np.zeros ((p1-p0, npair), dtype=int3c_ao.dtype)
    max_memory = mc_or_mc_grad.max_memory - lib.current_memory()[0]    
    blksize = int (min (max (max_memory * 1e6 / 8 / ((npair**2)*2), 20), 240))
    aux_loc = auxmol.ao_loc
    ao_ranges = balance_partition(aux_loc, blksize)
    for shl0, shl1, nL in ao_ranges:
        int3c_ao = get_int3c ((0, nbas, 0, nbas, shl0, shl1))  # (uv|P)
        p0, p1 = aux_loc[shl0], aux_loc[shl1]
        assert (int3c_ao.size == npair * (p1-p0))
        if ((int3c_ao.shape == (npair, p1-p0) and int3c_ao.flags.c_contiguous) or
            (int3c_ao.shape == (p1-p0, npair) and int3c_ao.flags.f_contiguous)):
            # I'm pretty sure I have to transpose for _ao2mo.nr_e2 to work
            if int3c_ao.flags.f_contiguous: int3c_ao = int3c_ao.T
            assert (int3c_ao.flags.c_contiguous)
            int3c_ao = lib.transpose (int3c_ao, out=buf)
        int3c[p0:p1] = _ao2mo.nr_e2(int3c_ao, mo_conc, mo_slice, aosym='s2', mosym=mosym, out=int3c[p0:p1])
        int3c_ao = None

    # (P|Q)
    int2c = linalg.cho_factor(auxmol.intor('int2c2e', aosym='s1'))
    
    # Solve (P|Q) d_Qij = (P|kl) d_ijkl
    dfcasdm2 = []
    for dm2 in casdm2:
        nmo_i, nmo_j = dm2.shape[0], dm2.shape[1]
        if mosym == 's2':
            # I'm not going to use the memory-efficient version because this is meant to be small
            dm2 = dm2.reshape ((-1, nmo0, nmo1))
            dm2 += dm2.transpose (0,2,1)
            diag_idx = numpy.arange(nmo0)
            diag_idx = diag_idx * (diag_idx+1) // 2 + diag_idx
            dm2 = lib.pack_tril (np.ascontiguousarray (dm2))
            dm2[:,diag_idx] *= 0.5
        dm2 = dm2.reshape (nmo_i*nmo_j, nmo_pair).T
        int3c_dm2 = np.dot (int3c, dm2)
        dfcasdm2.append (linalg.cho_solve (int2c, int3c_dm2).reshape (naux, nmo_i, nmo_j))

    return dfcasdm2



 
