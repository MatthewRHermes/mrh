# Moved to PySCF-Forge in the LASSCF migration (see MIGRATION.md). The gpu4mrh versions stay here.
# This module re-exports all of pyscf.mcscf.las.las_ao2mo (including private names) so that existing
# imports keep working.
import pyscf.mcscf.las.las_ao2mo as _forge
globals ().update ({k: v for k, v in vars (_forge).items () if not k.startswith ('__')})
from mrh.my_pyscf.df.sparse_df import sparsedf_array

#gpu accelerated version 
def get_h2eff_gpu (las,mo_coeff):
    from mrh.my_pyscf.gpu import libgpu
    log = lib.logger.new_logger (las, las.verbose)
    gpu=las.use_gpu
    nao, nmo = mo_coeff.shape
    count = 0
    ncore, ncas = las.ncore, las.ncas
    nocc = ncore + ncas
    mo_cas = mo_coeff[:,ncore:nocc]
    libgpu.push_mo_coeff(gpu,mo_coeff.copy(),mo_coeff.size)
    libgpu.extract_mo_cas(gpu,ncas, ncore, nao, nmo)
    naux = las.with_df.get_naoaux ()
    blksize = las.with_df.blockdim
    eri = 0
    t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
    eri1 = np.empty((nmo, int(ncas*ncas*(ncas+1)/2)),dtype='d')
    for cderi in las.with_df.loop (blksize=blksize):
        naux = cderi.shape[0]
        if las.verbose>=lib.logger.DEBUG and gpu:
            libgpu.get_h2eff_df_v1(gpu, cderi, nao, nmo, ncas, naux, ncore,eri1, count, id(las.with_df))
            bPmu = np.einsum('Pmn,nu->Pmu',lib.unpack_tril(cderi),mo_cas)
            bPvu = np.einsum('mv,Pmu->Pvu',mo_cas.conjugate(),bPmu)
            bumP = bPmu.transpose(2,1,0)
            buvP = bPvu.transpose(2,1,0)
            eri2 = np.einsum('uvP,wmP->uvwm', buvP, bumP)
            eri2 = np.einsum('mM,uvwm->Mwvu', mo_coeff.conjugate(),eri2)
            eri2 = lib.pack_tril (eri2.reshape (nmo*ncas, ncas, ncas)).reshape (nmo, -1)
            if np.allclose(eri1,eri2): log.debug("h2eff v1 is working")
            else: log.debug("h2eff v1 not working"); exit()
        elif gpu: 
            libgpu.get_h2eff_df_v1(gpu, cderi, nao, nmo, ncas, naux, ncore,eri1, count, id(las.with_df)); 
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
    from mrh.my_pyscf.gpu import libgpu
    log = lib.logger.new_logger (las, las.verbose)
    gpu=las.use_gpu
    nao, nmo = mo_coeff.shape
    count = 0
    ncore, ncas = las.ncore, las.ncas
    nocc = ncore + ncas
    mo_cas = mo_coeff[:,ncore:nocc]
    libgpu.push_mo_coeff(gpu,mo_coeff.copy(),mo_coeff.size)
    libgpu.init_eri_h2eff(gpu, nmo, ncas)
    libgpu.extract_mo_cas(gpu,ncas, ncore, nao, nmo)
    naux = las.with_df.get_naoaux ()
    blksize = las.with_df.blockdim
    eri =np.zeros((nmo,  int(ncas*ncas*(ncas+1)/2)))
    t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
    eri1 = np.zeros((nmo, int(ncas*ncas*(ncas+1)/2)),dtype='d')
    if las.verbose>lib.logger.DEBUG and gpu:
        eri_cpu = np.zeros((nmo, int(ncas*ncas*(ncas+1)/2)))
    for cderi in las.with_df.loop (blksize=blksize):
        #print(count)
        #t1 = lib.logger.timer (las, 'Sparsedf', *t0)
        naux = cderi.shape[0]
        if las.verbose>lib.logger.DEBUG and gpu:
            #print("using h2eff_df_v2")
            libgpu.get_h2eff_df_v2(gpu, cderi, nao, nmo, ncas, naux, ncore,eri1, count, id(las.with_df))
            bPmn = sparsedf_array (cderi)
            #print(lib.unpack_tril(cderi)[:3,:3,:3])
            #print(lib.unpack_tril(cderi).shape)
            print(mo_cas)
            bmuP1 = bPmn.contract1 (mo_cas)
            buvP = np.tensordot (mo_cas.conjugate (), bmuP1, axes=((0),(0)))
            eri2 = np.tensordot (bmuP1, buvP, axes=((2),(2)))
            eri2 = np.tensordot (mo_coeff.conjugate (), eri2, axes=((0),(0)))
            eri2 = lib.pack_tril (eri2.reshape (nmo*ncas, ncas, ncas)).reshape (nmo, -1)
            eri_cpu +=eri2
            #print(eri2)
        elif gpu: 
            libgpu.get_h2eff_df_v2(gpu, cderi, nao, nmo, ncas, naux, ncore,eri1, count, id(las.with_df)); 
        else: 
            bPmn = sparsedf_array (cderi)
            bmuP1 = bPmn.contract1 (mo_cas)
            buvP = np.tensordot (mo_cas.conjugate (), bmuP1, axes=((0),(0)))
            eri1 = np.tensordot (bmuP1, buvP, axes=((2),(2)))
            eri1 = np.tensordot (mo_coeff.conjugate (), eri1, axes=((0),(0)))
            eri1 = lib.pack_tril (eri1.reshape (nmo*ncas, ncas, ncas)).reshape (nmo, -1)
            cderi = bPmn = bmuP1 = buvP = None
        
        #t1 = lib.logger.timer (las, 'contract1 gpu', *t1)
        count+=1
    t0 = lib.logger.timer (las, 'las_ao2mo', *t0)
    libgpu.pull_eri_h2eff(gpu, eri, nmo, ncas)
    if las.verbose>lib.logger.DEBUG and gpu:
        if np.allclose(eri, eri_cpu): log.debug("h2eff_v2 working")
        else: 
            print("h2eff not working");
            diff = eri - eri_cpu
            non_zero_vals = np.nonzero(diff)
            print(non_zero_vals)
            print(np.max(np.abs(diff)))
            log.debug("h2eff not working");
            log.debug('eri_diff' + str(np.max(np.abs(eri-eri_cpu))));
            exit()
    eri1= None
    return eri
