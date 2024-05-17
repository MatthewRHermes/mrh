import numpy as np
from scipy import linalg
from pyscf import ao2mo, lib
from mrh.my_pyscf.df.sparse_df import sparsedf_array

def get_h2eff (las, mo_coeff=None):
    if mo_coeff is None: mo_coeff = las.mo_coeff
    nao, nmo = mo_coeff.shape
    ncore, ncas = las.ncore, las.ncas
    nocc = ncore + ncas
    mo_cas = mo_coeff[:,ncore:nocc]
    mo = [mo_coeff, mo_cas, mo_cas, mo_cas]
    if getattr (las, 'with_df', None) is not None:
        # Store intermediate with one contracted ao index for faster calculation of exchange!
        cderi = np.concatenate ([c for c in las.with_df.loop ()], axis=0)
        bPmn = sparsedf_array (cderi)
        bmuP = bPmn.contract1 (mo_cas)
        buvP = np.tensordot (mo_cas.conjugate (), bmuP, axes=((0),(0)))
        eri_muxy = np.tensordot (bmuP, buvP, axes=((2),(2)))
        eri = np.tensordot (mo_coeff.conjugate (), eri_muxy, axes=((0),(0)))
        eri = lib.pack_tril (eri.reshape (nmo*ncas, ncas, ncas)).reshape (nmo, -1)
        eri = lib.tag_array (eri, bmPu=bmuP.transpose (0,2,1))
        if las.verbose > lib.logger.DEBUG:
            eri_comp = las.with_df.ao2mo (mo, compact=True)
            lib.logger.debug(las,"CDERI two-step error: {}".format(linalg.norm(eri-eri_comp)))
    elif getattr (las._scf, '_eri', None) is not None:
        eri = ao2mo.incore.general (las._scf._eri, mo, compact=True)
    else:
        eri = ao2mo.outcore.general_iofree (las.mol, mo, compact=True)
    if eri.shape != (nmo,ncas*ncas*(ncas+1)//2):
        try:
            eri = eri.reshape (nmo, ncas*ncas*(ncas+1)//2)
        except ValueError as e:
            assert (nmo == ncas), str (e)
            eri = ao2mo.restore ('2kl', eri, nmo).reshape (nmo, ncas*ncas*(ncas+1)//2)
    return eri

def get_h2eff_slice (las, h2eff, idx, compact=None):
    ncas_cum = np.cumsum ([0] + las.ncas_sub.tolist ())
    i = ncas_cum[idx]
    j = ncas_cum[idx+1]
    ncore = las.ncore
    nocc = ncore + las.ncas
    eri = h2eff[ncore:nocc,:].reshape (las.ncas*las.ncas, -1)
    ix_i, ix_j = np.tril_indices (las.ncas)
    eri = eri[(ix_i*las.ncas)+ix_j,:]
    eri = ao2mo.restore (1, eri, las.ncas)[i:j,i:j,i:j,i:j]
    if compact: eri = ao2mo.restore (compact, eri, j-i)
    return eri

