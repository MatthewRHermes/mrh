import numpy as np
from scipy import linalg
from pyscf import ao2mo, lib
from mrh.my_pyscf.df.sparse_df import sparsedf_array

def get_h2eff_df (las, mo_coeff):
    # Store intermediate with one contracted ao index for faster calculation of exchange!
    nao, nmo = mo_coeff.shape
    ncore, ncas = las.ncore, las.ncas
    nocc = ncore + ncas
    mo_cas = mo_coeff[:,ncore:nocc]
    naux = las.with_df.get_naoaux ()
    mem_eris = 8*nmo*ncas*ncas*(ncas+1) // 2 // 1e6
    mem_av = las.max_memory - lib.current_memory ()[0] - mem_eris
    mem_int = 8*naux*ncas*nao // 1e6
    mem_enough_int = mem_av > mem_int
    if mem_enough_int:
        mem_av -= mem_int
        bmuP = []
    mem_per_aux = nao*(nao+1)//2 # cderi / bPmn
    mem_per_aux += nao*ncas # bmuP
    mem_per_aux += ncas*ncas # buvP
    mem_per_aux += nao*lib.num_threads () # wrk in contract1
    mem_per_aux *= 8 // 1e6
    mem_per_aux = max (1, mem_per_aux)
    blksize = max (1, min (naux, int (mem_av // mem_per_aux)))
    assert (blksize>1)
    eri = 0
    for cderi in las.with_df.loop (blksize=blksize):
        bPmn = sparsedf_array (cderi)
        bmuP1 = bPmn.contract1 (mo_cas)
        if mem_enough_int: bmuP.append (bmuP1)
        buvP = np.tensordot (mo_cas.conjugate (), bmuP1, axes=((0),(0)))
        eri1 = np.tensordot (bmuP1, buvP, axes=((2),(2)))
        eri1 = np.tensordot (mo_coeff.conjugate (), eri1, axes=((0),(0)))
        eri += lib.pack_tril (eri1.reshape (nmo*ncas, ncas, ncas)).reshape (nmo, -1)
    if mem_enough_int:
        eri = lib.tag_array (eri, bmPu=np.concatenate (bmuP, axis=-1).transpose (0,2,1))
    if las.verbose > lib.logger.DEBUG:
        eri_comp = las.with_df.ao2mo (mo, compact=True)
        lib.logger.debug(las,"CDERI two-step error: {}".format(linalg.norm(eri-eri_comp)))
    return eri

def get_h2eff (las, mo_coeff=None):
    if mo_coeff is None: mo_coeff = las.mo_coeff
    nao, nmo = mo_coeff.shape
    ncore, ncas = las.ncore, las.ncas
    nocc = ncore + ncas
    mo_cas = mo_coeff[:,ncore:nocc]
    mo = [mo_coeff, mo_cas, mo_cas, mo_cas]
    mem_eris = 8*nmo*ncas*ncas*(ncas+1) // 2 // 1e6
    mem_eris += 8*nao*ncas*ncas*ncas // 1e6 # intermediate
    mem_eris += 8*lib.num_threads ()*nao*nmo // 1e6 # intermediate
    mem_remaining = las.max_memory - lib.current_memory ()[0]
    if mem_eris > mem_remaining:
        raise MemoryError ("{} MB of {}/{} MB av/total for ERI array".format (
            mem_eris, mem_remaining, las.max_memory))
    if getattr (las, 'with_df', None) is not None:
        eri = get_h2eff_df (las, mo_coeff)
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

