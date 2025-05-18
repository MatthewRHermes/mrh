from pyscf import lo, lib
from pyscf.pbc import gto
from pyscf.tools import mo_mapping

# Active Space Guess for the PBC (Gamma-Point Only)
def mo_comps(aolabels_or_baslst, cell, mo_coeff, cart=False,
             orth_method='meta_lowdin'):
    '''For PBC: '''+mo_mapping.mo_comps.__doc__
    with lib.temporary_env(cell, cart=cart):
        assert (mo_coeff.shape[0] == cell.nao)
        s = cell.pbc_intor_symmetric('int1e_ovlp')
        lao = lo.orth.orth_ao(cell, orth_method, s=s)
        idx = gto.mole._aolabels2baslst(cell, aolabels_or_baslst)
        if len(idx) == 0:
            lib.logger.warn(cell, 'Required orbitals are not found')
        mo1 = reduce(np.dot, (lao[:,idx].T, s, mo_coeff))
        s1 = np.einsum('ki,ki->i', mo1, mo1)
    return s1
