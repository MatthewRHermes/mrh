import numpy as np
from scipy import linalg
from pyscf import ao2mo, lib
from mrh.my_pyscf.df.sparse_df import sparsedf_array
from mrh.my_pyscf.gpu import libgpu
DEBUG=False
def get_h2eff_df (las, mo_coeff):
    # Store intermediate with one contracted ao index for faster calculation of exchange!
    log = lib.logger.new_logger (las, las.verbose)
    gpu=las.use_gpu
    nao, nmo = mo_coeff.shape
    ncore, ncas = las.ncore, las.ncas
    nocc = ncore + ncas
    mo_cas = mo_coeff[:,ncore:nocc]
    #if gpu: 
    #    libgpu.libgpu_push_mo_coeff(gpu,mo_cas.copy(),mo_cas.size)
    naux = las.with_df.get_naoaux ()
    log.debug2 ("LAS DF ERIs: %d MB used of %d MB total available", lib.current_memory ()[0], las.max_memory)
    mem_eris = 8*(nao+nmo)*ncas*ncas*ncas / 1e6
    mem_eris += 8*lib.num_threads ()*nao*nmo / 1e6 
    mem_av = las.max_memory - lib.current_memory ()[0] - mem_eris
    mem_int = 16*naux*ncas*nao / 1e6
    mem_enough_int = mem_av > mem_int
    t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
    if mem_enough_int:
        mem_av -= mem_int
        bmuP = []
        log.debug ("LAS DF ERI including intermediate cache")
    else:
        log.debug ("LAS DF ERI not including intermediate cache")
    safety_factor = 1.1
    mem_per_aux = nao*ncas # bmuP
    mem_per_aux += ncas*ncas # buvP
    mem_per_aux += nao*lib.num_threads () # wrk in contract1
    if not isinstance (getattr (las.with_df, '_cderi', None), np.ndarray):
        mem_per_aux += 3*nao*(nao+1)//2 # cderi / bPmn
        # NOTE: I think a linalg.norm operation in sparsedf_array might be doubling the memory
        # footprint of bPmn below
    else:
        mem_per_aux += nao*(nao+1) # see note above
    mem_per_aux *= safety_factor * 8 / 1e6
    mem_per_aux = max (1, mem_per_aux)
    blksize = max (1, min (naux, int (mem_av / mem_per_aux)))
    assert (blksize>1)
    log.debug2 ("LAS DF ERI blksize = %d, mem_av = %d MB, mem_per_aux = %d MB", blksize, mem_av, mem_per_aux)
    log.debug2 ("LAS DF ERI naux = %d, nao = %d, nmo = %d", naux, nao, nmo)
    eri = 0
    for cderi in las.with_df.loop (blksize=blksize):
        #t1 = lib.logger.timer (las, 'Sparsedf', *t0)
        bPmn = sparsedf_array (cderi)
        bmuP1 = bPmn.contract1 (mo_cas)
        #t1 = lib.logger.timer (las, 'contract1', *t1)
        log.debug2 ("LAS DF ERI bPmn shape = %s; shares memory? %s %s; C_CONTIGUOUS? %s",
                 str (bPmn.shape), str (np.shares_memory (bPmn, cderi)),
                 str (np.may_share_memory (bPmn, cderi)),
                 str (bPmn.flags['C_CONTIGUOUS']))
        if mem_enough_int : bmuP.append (bmuP1)
        buvP = np.tensordot (mo_cas.conjugate (), bmuP1, axes=((0),(0)))
        eri1 = np.tensordot (bmuP1, buvP, axes=((2),(2)))
        eri1 = np.tensordot (mo_coeff.conjugate (), eri1, axes=((0),(0)))
        eri += lib.pack_tril (eri1.reshape (nmo*ncas, ncas, ncas)).reshape (nmo, -1)
        cderi = bPmn = bmuP1 = buvP = eri1 = None
        #t1 = lib.logger.timer (las, 'rest of the calculation', *t1)
    #if mem_enough_int and not gpu: eri = lib.tag_array (eri, bmPu=np.concatenate (bmuP, axis=-1).transpose (0,2,1))
    if mem_enough_int : eri = lib.tag_array (eri, bmPu=np.concatenate (bmuP, axis=-1).transpose (0,2,1))
    if las.verbose > lib.logger.DEBUG:
        eri_comp = las.with_df.ao2mo (mo_coeff, compact=True)
        eri_comp = eri_comp[:,ncore:nocc,ncore:nocc,ncore:nocc]
        eri_comp = lib.pack_tril (eri_comp.reshape (nmo*ncas, ncas, ncas)).reshape (nmo, -1)
        lib.logger.debug(las,"CDERI two-step error: {}".format(linalg.norm(eri-eri_comp)))
    t0 = lib.logger.timer (las, 'las_ao2mo', *t0)
    return eri

#gpu accelerated version 
def get_h2eff_gpu (las,mo_coeff):
    log = lib.logger.new_logger (las, las.verbose)
    gpu=las.use_gpu
    nao, nmo = mo_coeff.shape
    count = 0
    ncore, ncas = las.ncore, las.ncas
    nocc = ncore + ncas
    mo_cas = mo_coeff[:,ncore:nocc]
    libgpu.libgpu_push_mo_coeff(gpu,mo_coeff.copy(),mo_coeff.size)
    libgpu.libgpu_extract_mo_cas(gpu,ncas, ncore, nao)
    naux = las.with_df.get_naoaux ()
    blksize = las.with_df.blockdim
    eri = 0
    t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
    eri1 = np.empty((nmo, int(ncas*ncas*(ncas+1)/2)),dtype='d')
    for cderi in las.with_df.loop (blksize=blksize):
        naux = cderi.shape[0]
        if DEBUG and gpu:
            libgpu.libgpu_get_h2eff_df_v1(gpu, cderi, nao, nmo, ncas, naux, ncore,eri1, count, id(las.with_df))
            bPmu = np.einsum('Pmn,nu->Pmu',lib.unpack_tril(cderi),mo_cas)
            bPvu = np.einsum('mv,Pmu->Pvu',mo_cas.conjugate(),bPmu)
            bumP = bPmu.transpose(2,1,0)
            buvP = bPvu.transpose(2,1,0)
            eri2 = np.einsum('uvP,wmP->uvwm', buvP, bumP)
            eri2 = np.einsum('mM,uvwm->Mwvu', mo_coeff.conjugate(),eri2)
            eri2 = lib.pack_tril (eri2.reshape (nmo*ncas, ncas, ncas)).reshape (nmo, -1)
            if np.allclose(eri1,eri2): print("h2eff v1 is working")
            else: print("h2eff v1 not working"); exit()
        elif gpu: libgpu.libgpu_get_h2eff_df_v1(gpu, cderi, nao, nmo, ncas, naux, ncore,eri1, count, id(las.with_df)); 
        else: 
            bPmn = sparsedf_array (cderi)
            bmuP1 = bPmn.contract1 (mo_cas)
            buvP = np.tensordot (mo_cas.conjugate (), bmuP1, axes=((0),(0)))
            eri1 = np.tensordot (bmuP1, buvP, axes=((2),(2)))
            eri1 = np.tensordot (mo_coeff.conjugate (), eri1, axes=((0),(0)))
            eri1 = lib.pack_tril (eri1.reshape (nmo*ncas, ncas, ncas)).reshape (nmo, -1)
            cderi = bPmn = bmuP1 = buvP = None
        count+=1
        eri +=eri1
    t0 = lib.logger.timer (las, 'las_ao2mo', *t0)
    eri1= None
    return eri

#even faster gpu accelerated version currently, currently not working. 
def get_h2eff_gpu_v2 (las,mo_coeff):
    log = lib.logger.new_logger (las, las.verbose)
    gpu=las.use_gpu
    nao, nmo = mo_coeff.shape
    count = 0
    ncore, ncas = las.ncore, las.ncas
    nocc = ncore + ncas
    mo_cas = mo_coeff[:,ncore:nocc]
    libgpu.libgpu_push_mo_coeff(gpu,mo_coeff.copy(),mo_coeff.size)
    libgpu.libgpu_init_eri_h2eff(gpu, nmo, ncas)
    libgpu.libgpu_extract_mo_cas(gpu,ncas, ncore, nao)
    naux = las.with_df.get_naoaux ()
    blksize = las.with_df.blockdim
    eri =np.zeros((nmo,  int(ncas*ncas*(ncas+1)/2)))
    t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
    eri1 = np.zeros((nmo, int(ncas*ncas*(ncas+1)/2)),dtype='d')
    if DEBUG and gpu:
        eri_cpu = np.zeros((nmo, int(ncas*ncas*(ncas+1)/2)))
    for cderi in las.with_df.loop (blksize=blksize):
        t1 = lib.logger.timer (las, 'Sparsedf', *t0)
        naux = cderi.shape[0]
        if DEBUG and gpu:
            libgpu.libgpu_get_h2eff_df_v2(gpu, cderi, nao, nmo, ncas, naux, ncore,eri1, count, id(las.with_df))
            bPmn = sparsedf_array (cderi)
            bmuP1 = bPmn.contract1 (mo_cas)
            buvP = np.tensordot (mo_cas.conjugate (), bmuP1, axes=((0),(0)))
            eri2 = np.tensordot (bmuP1, buvP, axes=((2),(2)))
            eri2 = np.tensordot (mo_coeff.conjugate (), eri2, axes=((0),(0)))
            eri2 = lib.pack_tril (eri2.reshape (nmo*ncas, ncas, ncas)).reshape (nmo, -1)
            eri_cpu +=eri2
        elif gpu: 
            libgpu.libgpu_get_h2eff_df_v2(gpu, cderi, nao, nmo, ncas, naux, ncore,eri1, count, id(las.with_df)); 
        else: 
            bPmn = sparsedf_array (cderi)
            bmuP1 = bPmn.contract1 (mo_cas)
            buvP = np.tensordot (mo_cas.conjugate (), bmuP1, axes=((0),(0)))
            eri1 = np.tensordot (bmuP1, buvP, axes=((2),(2)))
            eri1 = np.tensordot (mo_coeff.conjugate (), eri1, axes=((0),(0)))
            eri1 = lib.pack_tril (eri1.reshape (nmo*ncas, ncas, ncas)).reshape (nmo, -1)
            cderi = bPmn = bmuP1 = buvP = None
        
        t1 = lib.logger.timer (las, 'contract1 gpu', *t1)
        count+=1
    libgpu.libgpu_pull_eri_h2eff(gpu, eri, nmo, ncas)
    if DEBUG and gpu:
        if np.allclose(eri, eri_cpu): print("h2eff_v2 working")
        else: print("h2eff not working");print('eri_diff'); print(np.max(np.abs(eri-eri_cpu)));exit()#print('eri_cpu'); print(eri_cpu);exit()
    eri1= None
    return eri




def get_h2eff (las, mo_coeff=None):
    if mo_coeff is None: mo_coeff = las.mo_coeff
    nao, nmo = mo_coeff.shape
    ncore, ncas = las.ncore, las.ncas
    nocc = ncore + ncas
    mo_cas = mo_coeff[:,ncore:nocc]
    mo = [mo_coeff, mo_cas, mo_cas, mo_cas]
    mem_eris = 8*(nao+nmo)*ncas*ncas*ncas // 1e6
    mem_eris += 8*lib.num_threads ()*nao*nmo // 1e6 # intermediate
    mem_remaining = las.max_memory - lib.current_memory ()[0]
    if mem_eris > mem_remaining:
        raise MemoryError ("{} MB of {}/{} MB av/total for ERI array".format (
            mem_eris, mem_remaining, las.max_memory))
    if getattr (las, 'with_df', None) is not None:
        if las.use_gpu: eri = get_h2eff_gpu (las,mo_coeff)#the full version is not working yet.
        else: eri = get_h2eff_df (las, mo_coeff)
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

def get_h2cas (las, mo_coeff=None):
    ''' Get the 2-electron integral of the active orbitals only '''
    if mo_coeff is None: mo_coeff = las.mo_coeff
    nao, nmo = mo_coeff.shape
    ncore, ncas = las.ncore, las.ncas
    nocc = ncore + ncas
    mo_cas = mo_coeff[:,ncore:nocc]
    if getattr (las, 'with_df', None) is not None:
        eri = las.with_df.ao2mo (mo_cas)
    elif getattr (las._scf, '_eri', None) is not None:
        eri = ao2mo.full (las._scf._eri, mo_cas, max_memory=las.max_memory)
    else:
        eri = ao2mo.full (las.mol, mo_cas, verbose=las.verbose, max_memory=las.max_memory)
    return ao2mo.restore (1, eri, ncas)



