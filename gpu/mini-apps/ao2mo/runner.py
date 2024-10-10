gpu_run=1
N=0
if gpu_run:from mrh.my_pyscf.gpu import libgpu
import pyscf
import numpy 
if gpu_run:from gpu4pyscf import patch_pyscf
from geometry_generator import generator
from pyscf import gto, scf, tools, mcscf, lib
from mrh.my_pyscf.mcscf.lasscf_async import LASSCF
from pyscf.mcscf import avas	
if gpu_run:gpu = libgpu.libgpu_init()
lib.logger.TIMER_LEVEL=lib.logger.INFO
nfrags=1;basis='sto3g';
if N:
    atom='''Li 0.0 0.0 0.0;
    Li 0.0 0.0 1.0'''
    mol=gto.M(atom=atom,basis=basis,verbose=4)
    if gpu_run:mol=gto.M(use_gpu=gpu, atom=atom,basis=basis)#,verbose=4, output=output_file)
else:
    if gpu_run:mol=gto.M(use_gpu=gpu, atom=generator(nfrags),basis=basis)#,verbose=4,output=outputfile,max_memory=160000)
mf=scf.RHF(mol)
mf=mf.density_fit(auxbasis='weigend')
mf.run()
mc=mcscf.CASSCF(mf, nfrags*2, nfrags*2)
with_df = mc._scf.with_df

def init_eri_gpu_v0 (mo, casscf, with_df):
    nao, nmo = mo.shape
    ncore = casscf.ncore
    ncas = casscf.ncas
    nocc = ncore + ncas
    naoaux = with_df.get_naoaux()
    mo = numpy.asarray(mo, order='F')
    fxpp = numpy.empty ((nmo, nmo, naoaux))
    blksize = with_df.blockdim
    bufpa = numpy.empty((naoaux,nmo,ncas))
    bufs1 = numpy.empty((blksize,nmo,nmo))
    fxpp_keys = []
    b0 = 0
    j_pc = numpy.zeros((nmo,ncore))
    k_pc = numpy.zeros((nmo,ncore))
    k_cp = numpy.zeros((ncore,nmo))
    if gpu: 
        libgpu.libgpu_push_mo_coeff(gpu, mo, nao*nmo)
        libgpu.libgpu_init_jk_ao2mo(gpu, ncore, nmo) 
    count = 0
    for k, eri1 in enumerate(with_df.loop(blksize)):
        naux = eri1.shape[0]
        bufpp = numpy.empty((nmo, nmo, naux))
        bufpa_slice = numpy.empty((naux, nmo, ncas))
        libgpu.libgpu_df_ao2mo_pass1(gpu,naux,nmo,nao,ncore,ncas,bufpp,bufpa_slice,eri1,k,id(with_df))
        fxpp_keys.append([k, b0, b0+naux])
        fxpp[:,:,b0:b0+naux] = bufpp #bufpp is already transposed in GPU kernel
        bufpa[b0:b0+naux] = bufpa_slice
        b0 += naux
    if gpu: 
        libgpu.libgpu_pull_jk_ao2mo (gpu, j_pc, k_cp, nmo, ncore)
    k_pc = k_cp.T.copy()
    print("finishing v0")
    return fxpp,bufpa,j_pc, k_pc
     
def init_eri_gpu_v1 (mo, casscf, with_df):
    nao, nmo = mo.shape
    ncore = casscf.ncore
    ncas = casscf.ncas
    nocc = ncore + ncas
    naoaux = with_df.get_naoaux()
    mo = numpy.asarray(mo, order='F')
    fxpp = numpy.empty ((nmo, nmo, naoaux))
    blksize = with_df.blockdim
    bufpa = numpy.empty((naoaux,nmo,ncas))
    bufs1 = numpy.empty((blksize,nmo,nmo))
    fxpp_keys = []
    b0 = 0
    j_pc = numpy.zeros((nmo,ncore))
    k_pc = numpy.zeros((nmo,ncore))
    k_cp = numpy.zeros((ncore,nmo))

    if gpu:
        arg = numpy.array([-1, -1, -1, -1], dtype=numpy.int32)
        libgpu.libgpu_get_dfobj_status(gpu, id(with_df), arg)
        if arg[2] > -1: load_eri = False
    libgpu.libgpu_push_mo_coeff(gpu, mo, nao*nmo)
    libgpu.libgpu_init_jk_ao2mo(gpu, ncore, nmo) # initializes j_pc and k_pc to be pulled
    libgpu.libgpu_init_ints_ao2mo(gpu, naoaux, nmo, ncas) #initializes fxpp and bufpa on pinned memory
    count = 0
    bufpp = numpy.empty((nmo, nmo, blksize))
    bufpa_slice = numpy.empty((blksize, nmo, ncas))
    for k, eri1 in enumerate(with_df.loop(blksize)):
        naux = eri1.shape[0]
        fxpp_keys.append([k, b0, b0+naux])
    #for k, eri1 in enumerate(with_df.loop(blksize)): 
    #for count in range(arg[2]):
    for count in range(k+1):
        #arg = numpy.array([-1, -1, count, -1], dtype = numpy.int32)
        #libgpu.libgpu_get_dfobj_status(gpu, id(with_df),arg)
        #naux = arg[0]
        libgpu.libgpu_df_ao2mo_pass1_v2(gpu,blksize,nmo,nao,ncore,ncas,naux,eri1,count,id(with_df)) # includes pulling fxpp and bufpa
    libgpu.libgpu_pull_jk_ao2mo (gpu, j_pc, k_cp, nmo, ncore)
    libgpu.libgpu_pull_ints_ao2mo(gpu, fxpp, bufpa, naoaux, nmo, ncas)
    k_pc = k_cp.T.copy()
    print("finishing v1")
    return fxpp,bufpa, j_pc, k_pc
 
fxpp, bufpa, j_pc, k_pc = init_eri_gpu_v0 (mf.mo_coeff, mc, with_df)
fxpp2, bufpa2, j_pc2, k_pc2 = init_eri_gpu_v1 (mf.mo_coeff, mc, with_df)
print("fxpp check", numpy.allclose(fxpp,fxpp2))
print(fxpp-fxpp2)
print(fxpp2)
print("bufpa check", numpy.allclose(bufpa, bufpa2))
print("j_pc check", numpy.allclose(j_pc, j_pc2))
print("k_pc check", numpy.allclose(k_pc, k_pc2))
