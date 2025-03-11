#miniapp to debug and test performance of h2eff versions
'''
all versions do the following
   b_{u1a2}^P = b^P_{u1u2}m^{a2}_{u2}           ---- 1
   b_{a1a2}^P = b^P_{u1a2}m^{a1}_{u1}           ---- 2
   g^{u1a2}_{a3a4} = b^P_{p1a2}b^P_{a3a4}       ---- 3
   g^{p1a2}_{a3a4} = g^{u1a2}_{a3a4}m^{p1}_{u1} ---- 4 
   
  
   gpu does 1, 2, 3 and 4 using pageable memory and with fewer initializations
   gpu_v1 does 1, 2, 3 and 4 using pageable memory 
   gpu_v2 does 1, 2, 3 and 4 using pageable memory (not yet working) 

''' 


gpu_run=1
DEBUG=1
import time
if gpu_run:from mrh.my_pyscf.gpu import libgpu
import pyscf
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
if gpu_run:from gpu4mrh import patch_pyscf
from mrh.tests.gpu.geometry_generator import generator
from pyscf import gto, scf, tools, mcscf, lib
from mrh.my_pyscf.mcscf.lasscf_async import LASSCF
from pyscf import ao2mo, lib
from mrh.my_pyscf.df.sparse_df import sparsedf_array
from pyscf.mcscf import avas
if gpu_run:gpu = libgpu.libgpu_init()
lib.logger.TIMER_LEVEL=lib.logger.INFO
nfrags=6;basis='sto3g';
outputfile=str(nfrags)+'_'+str(basis)+'_out_cpu_ref_2.log';
if gpu_run:outputfile=str(nfrags)+'_'+str(basis)+'_out_gpu_ref_1_debug.log';
mol=gto.M(atom=generator(nfrags),basis=basis,verbose=4,output=outputfile,max_memory=160000)
if gpu_run:mol=gto.M(use_gpu=gpu, atom=generator(nfrags),basis=basis,verbose=4,output=outputfile,max_memory=160000)
mf=scf.RHF(mol)
mf=mf.density_fit()#auxbasis='weigend')
mf.run()
las=LASSCF(mf, list((2,)*nfrags),list((2,)*nfrags),verbose=4)#, use_gpu=gpu)
if gpu_run:las=LASSCF(mf, list((2,)*4),list((2,)*4), use_gpu=gpu,verbose=4)
frag_atom_list=[list(range(1+4*nfrag,3+4*nfrag)) for nfrag in range(4)]
ncas,nelecas,guess_mo_coeff=avas.kernel(mf, ["C 2pz"])
mo_coeff=las.set_fragments_(frag_atom_list, guess_mo_coeff)

def get_h2eff_df (las, mo_coeff):
    # Store intermediate with one contracted ao index for faster calculation of exchange!
    log = lib.logger.new_logger (las, las.verbose)
    gpu=las.use_gpu
    nao, nmo = mo_coeff.shape
    ncore, ncas = las.ncore, las.ncas
    nocc = ncore + ncas
    mo_cas = mo_coeff[:,ncore:nocc]
    if gpu: 
        libgpu.libgpu_push_mo_coeff(gpu,mo_cas.copy(),mo_cas.size)
    naux = las.with_df.get_naoaux ()
    log.debug2 ("LAS DF ERIs: %d MB used of %d MB total available", lib.current_memory ()[0], las.max_memory)
    mem_eris = 8*(nao+nmo)*ncas*ncas*ncas / 1e6
    mem_eris += 8*lib.num_threads ()*nao*nmo / 1e6 
    mem_av = las.max_memory - lib.current_memory ()[0] - mem_eris
    mem_int = 16*naux*ncas*nao / 1e6
    mem_enough_int = mem_av > mem_int
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
    t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
    for cderi in las.with_df.loop (blksize=blksize):
        t1 = lib.logger.timer (las, 'Sparsedf', *t0)
        bPmn = sparsedf_array (cderi)
        bmuP1 = bPmn.contract1 (mo_cas)
        t1 = lib.logger.timer (las, 'contract1', *t1)
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
        t1 = lib.logger.timer (las, 'rest of the calculation', *t1)
    if mem_enough_int and not gpu: eri = lib.tag_array (eri, bmPu=np.concatenate (bmuP, axis=-1).transpose (0,2,1))
    if las.verbose > lib.logger.DEBUG:
        eri_comp = las.with_df.ao2mo (mo_coeff, compact=True)
        eri_comp = eri_comp[:,ncore:nocc,ncore:nocc,ncore:nocc]
        eri_comp = lib.pack_tril (eri_comp.reshape (nmo*ncas, ncas, ncas)).reshape (nmo, -1)
        lib.logger.debug(las,"CDERI two-step error: {}".format(linalg.norm(eri-eri_comp)))
    return eri

def get_h2eff_gpu (las,mo_coeff):
    log = lib.logger.new_logger (las, las.verbose)
    gpu=las.use_gpu
    nao, nmo = mo_coeff.shape
    count = 0
    ncore, ncas = las.ncore, las.ncas
    nocc = ncore + ncas
    mo_cas = mo_coeff[:,ncore:nocc]
    libgpu.libgpu_push_mo_coeff(gpu,mo_coeff.copy(),mo_coeff.size)
    naux = las.with_df.get_naoaux ()
    blksize = las.with_df.blockdim
    eri = 0
    t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
    eri1 = np.empty((nmo, int(ncas*ncas*(ncas+1)/2)),dtype='d')
    for cderi in las.with_df.loop (blksize=blksize):
        t1 = lib.logger.timer (las, 'Sparsedf', *t0)
        naux = cderi.shape[0]
        if DEBUG and gpu:
            libgpu.libgpu_get_h2eff_df(gpu, cderi, nao, nmo, ncas, naux, ncore,eri1, count, id(las.with_df))
            bPmu = np.einsum('Pmn,nu->Pmu',lib.unpack_tril(cderi),mo_cas)
            bPvu = np.einsum('mv,Pmu->Pvu',mo_cas.conjugate(),bPmu)
            bumP = bPmu.transpose(2,1,0)
            buvP = bPvu.transpose(2,1,0)
            eri2 = np.einsum('uvP,wmP->uvwm', buvP, bumP)
            eri2 = np.einsum('mM,uvwm->Mwvu', mo_coeff.conjugate(),eri2)
            eri2 = lib.pack_tril (eri2.reshape (nmo*ncas, ncas, ncas)).reshape (nmo, -1)
            if np.allclose(eri1,eri2): print("h2eff is working")
            else: print("h2eff not working"); exit()
        elif gpu: libgpu.libgpu_get_h2eff_df(gpu, cderi, nao, nmo, ncas, naux, ncore,eri1, count, id(las.with_df)); 
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
        eri +=eri1
    eri1= None
    return eri
def get_h2eff_gpu_v1 (las,mo_coeff):
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
    eri = 0
    t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
    eri1 = np.empty((nmo, int(ncas*ncas*(ncas+1)/2)),dtype='d')
    for cderi in las.with_df.loop (blksize=blksize):
        t1 = lib.logger.timer (las, 'Sparsedf', *t0)
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
        t1 = lib.logger.timer (las, 'contract1 gpu', *t1)
        count+=1
        eri +=eri1
    eri1= None
    return eri


def get_h2eff_gpu_v2 (las,mo_coeff):
    log = lib.logger.new_logger (las, las.verbose)
    gpu=las.use_gpu
    nao, nmo = mo_coeff.shape
    print(nao,nmo)
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
        print(naux)
        if DEBUG and gpu:
            libgpu.libgpu_get_h2eff_df_v2(gpu, cderi, nao, nmo, ncas, naux, ncore,eri1, count, id(las.with_df))
            bPmn = sparsedf_array (cderi)
            bmuP1 = bPmn.contract1 (mo_cas)
            buvP = np.tensordot (mo_cas.conjugate (), bmuP1, axes=((0),(0)))
            eri2 = np.tensordot (bmuP1, buvP, axes=((2),(2)))
            eri2 = np.tensordot (mo_coeff.conjugate (), eri2, axes=((0),(0)))
            eri2 = lib.pack_tril (eri2.reshape (nmo*ncas, ncas, ncas)).reshape (nmo, -1)
            print(eri2)
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
    libgpu.libgpu_pull_eri_h2eff(gpu, eri, nao, ncas)
    if DEBUG and gpu:
        if np.allclose(eri, eri_cpu): print("h2eff_v2 working")
        else: print("h2eff not working");print('eri_diff'); print(np.max(np.abs(eri-eri_cpu)));exit()#print('eri_cpu'); print(eri_cpu);exit()
    eri1= None
    return eri



#if gpu_run:gpu = libgpu.libgpu_init()

t0=time.time()
for _ in range(1):  get_h2eff_gpu_v1 (las, mf.mo_coeff)
t1=time.time()
if gpu_run:libgpu.libgpu_destroy_device(gpu)
