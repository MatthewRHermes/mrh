''' 
This miniapp is for development of gpu accelerated version of creation of impurity subspace cholesky vectors by the following equations

     
     b^P_{e_{K_1}\mu_2} = b^P_{\mu_1\mu_2}M^{\mu_1}_{e_{K_1}} 
     b^P_{e_{K_1}e_{K_2}} = b^P_{e_{K_1}\mu_2}M^{\mu_2}_{e_{K_2}} 

'''




gpu_run=1
import numpy as np
import pyscf
from pyscf import ao2mo, lib, df, gto, scf
from pyscf.mcscf import avas
from mrh.tests.gpu.geometry_generator import generator
from mrh.my_pyscf.mcscf.lasscf_async import LASSCF
DEBUG=1 
TIMING=0 #if true, runs both cpu version and gpu version n times and reports wall times 
if gpu_run: 
    import gpu4mrh
    from gpu4mrh import patch_pyscf
    from mrh.my_pyscf.gpu import libgpu
    gpu=libgpu.init()

def impham_cpu_original(self, imporb_coeff):
    mf = self._scf
    nimp = imporb_coeff.shape[1]
    _cderi=np.empty((mf.with_df.get_naoaux(), nimp*(nimp+1)//2),dtype=imporb_coeff.dtype)
    blksize=240
    b0=0
    ijmosym, mij_pair, moij, ijslice = ao2mo.incore._conc_mos (imporb_coeff, imporb_coeff,compact=True)
    for eri1 in mf.with_df.loop(blksize=blksize):
        b1=b0+eri1.shape[0]
        eri2=_cderi[b0:b1]
        eri2 = ao2mo._ao2mo.nr_e2 (eri1, moij, ijslice, aosym='s2', mosym=ijmosym,out=eri2)
        b0 = b1
    #return lib.unpack_tril(_cderi)
    #print(lib.unpack_tril(_cderi)[:2])
    #print(_cderi[:-2])
    return _cderi
       
def impham_cpu_naive(self, imporb_coeff):
    mf = self._scf
    nimp = imporb_coeff.shape[1]
    _cderi=np.empty((mf.with_df.get_naoaux(), nimp, nimp),dtype=imporb_coeff.dtype)
    blksize=240
    b0=0
    for eri1 in mf.with_df.loop(blksize=blksize):
        b1=b0+eri1.shape[0]
        eri_up = lib.unpack_tril(eri1)
        _cderi[b0:b1]=np.einsum('pIj,jJ->pIJ',np.einsum('pij,iI->pIj',eri_up,imporb_coeff),imporb_coeff)
        b0 = b1
    return _cderi

def impham_gpu_v1(self, imporb_coeff):
    
    mf=self._scf
    nao_s,nao_f = imporb_coeff.shape
    naoaux = mf.with_df.get_naoaux()
    blksize=mf.with_df.blockdim
    #_cderi=np.empty((naoaux, nao_f, nao_f),dtype=imporb_coeff.dtype)
    _cderi=np.empty((naoaux, nao_f*(nao_f+1)//2),dtype=imporb_coeff.dtype)
    libgpu.push_mo_coeff(gpu, imporb_coeff, nao_s*nao_f)
    libgpu.init_eri_impham(gpu, naoaux, nao_f)
    for k, eri1 in enumerate(mf.with_df.loop(blksize)):pass;
    for count in range(k+1): 
        arg = np.array([-1, -1, count, -1], dtype = np.int32)
        libgpu.get_dfobj_status(gpu, id(mf.with_df),arg)
        naux = arg[0]
        libgpu.compute_eri_impham (gpu, nao_s, nao_f, blksize, naux, count, id(mf.with_df))
    libgpu.pull_eri_impham(gpu, _cderi, naoaux, nao_f)  
    return _cderi

nfrags=8;basis='631g';
nimp=44
atom=generator(nfrags)
if gpu_run:mol=gto.M(use_gpu=gpu, atom=atom,basis=basis)
else: mol=gto.M(atom=atom,basis=basis)
mf=scf.RHF(mol)
mf=mf.density_fit().newton()
mf.max_cycle=1
mf.run()

def tester (las, imporb_coeff):
    cderi_original = impham_cpu_original(las,imporb_coeff) 
    cderi_gpu_v1 = impham_gpu_v1(las,imporb_coeff) 
    if not (np.allclose(cderi_original,cderi_gpu_v1)):
        diff =np.abs(cderi_original-cderi_gpu_v1)
        #print((cderi_original-cderi_gpu_v1)[:-2])
    #print(cderi_gpu_v1)
        idx = np.unravel_index(np.argmax(diff),diff.shape)
        print(np.max(diff), idx, diff.shape, cderi_original[idx],cderi_gpu_v1[idx], np.average(diff))
    #print(np.max(diff), np.unravel_index(np.argmax(diff),diff.shape, ))
        exit()
    else:
        print("we are done!!! commit and good night")


if gpu_run:las=LASSCF(mf, list((2,)*nfrags),list((2,)*nfrags), use_gpu=gpu,verbose=4)
frag_atom_list=[list(range(1+4*nfrag,3+4*nfrag)) for nfrag in range(nfrags)]
ncas,nelecas,guess_mo_coeff=avas.kernel(mf, ["C 2pz"])
mo_coeff=las.set_fragments_(frag_atom_list, guess_mo_coeff)

nao, nmo = mf.mo_coeff.shape
#print(mf.with_df.get_naoaux(), nao, nimp)
imporb_coeff=np.random.random((nao, nimp)).astype(np.float64)-.5#,dtype=np.float64)

tester (las, np.random.random((nao, nimp)).astype(np.float64)-.5)
tester (las, np.random.random((nao, 5)).astype(np.float64)-.5)
#tester (las, np.random.random((nao, nimp//3)).astype(np.float64)-.5)
#tester (las, np.random.random((nao, nimp*2)).astype(np.float64)-.5)

if TIMING:
    n=15
    import time
    t0 = time.time()
    for _ in range(n): cderi_original = impham_cpu_original(las,imporb_coeff) 
    t1 = time.time()
    for _ in range(n): cderi_gpu_v1 = impham_gpu_v1(las,imporb_coeff) 
    t2 = time.time()
    print("CPU time = ",round(t1-t0,2)," seconds")
    print("GPU time = ",round(t2-t1,2)," seconds")

 
