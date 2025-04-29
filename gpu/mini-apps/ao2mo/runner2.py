#miniapp to debug and test performance of ao2mo versions
'''
all versions do the following
   b_{u1p2}^P = b^P_{u1u2}m^{p2}_{u2}        ---- 1
   b_{p1p2}^P = b^P_{u1p2}m^{p1}_{u1}        ---- 2
   g^{p1p2}_{a1a2} = b^P_{p1p2}b^P_{a1a2}    ---- 3
   g^{p1a1}_{p2a2} = b^P_{p1a1}b^P_{p2a2}    ---- 4
   
  
   gpu_v3 does 1, 2 and 3 and pulls back g^{p1p2}_{a1a2} and b^P_{p1a1}
   gpu_v4 does 1, 2, 3 and 4 and pulls back g^{p1p2}_{a1a2} and g^{p1a1}_{p2a2} 

''' 

gpu_run=1
if gpu_run:from mrh.my_pyscf.gpu import libgpu
import pyscf
import numpy 
import ctypes
if gpu_run:from gpu4mrh import patch_pyscf
from mrh.tests.gpu.geometry_generator import generator
from pyscf import gto, scf, tools, mcscf, lib
from mrh.my_pyscf.mcscf.lasscf_async import LASSCF
from pyscf.mcscf import avas	
from pyscf.ao2mo import _ao2mo
import time
DEBUG = True
PERFORMANCE = True
nruns=20
if gpu_run:gpu = libgpu.libgpu_init()
lib.logger.TIMER_LEVEL=lib.logger.INFO

nfrags=4;basis='ccpvtz';
if gpu_run:mol=gto.M(use_gpu=gpu, atom=generator(nfrags),basis=basis)#,verbose=4,output=outputfile,max_memory=160000)
#else: mol=gto.M(atom=generator(nfrags),basis=basis)
mf=scf.RHF(mol)
mf=mf.density_fit()#auxbasis='weigend')
mf.run()
mc=mcscf.CASSCF(mf, nfrags*2, nfrags*2)
with_df = mc._scf.with_df
nao, nmo = mc.mo_coeff.shape
ncore = mc.ncore
ncas = mc.ncas
nocc = ncore + ncas
naoaux = with_df.get_naoaux()
blksize = with_df.blockdim
print("nao: ",mol.nao, " naux: ",mf.with_df.get_naoaux(), "blksize: ",blksize )

def prange(start, end, step):
    for i in range(start, end, step):
        yield i, min(i+step, end)

def init_eri_cpu (mo, casscf, with_df):
    mo = numpy.asarray(mo, order='F')
    fxpp = numpy.empty ((nmo, nmo, naoaux))
    bufpa = numpy.empty((naoaux,nmo,ncas))
    fxpp_keys = []
    b0 = 0
    j_pc_cpu = numpy.zeros((nmo,ncore))
    k_pc_cpu = numpy.zeros((nmo,ncore))
    k_cp = numpy.zeros((ncore,nmo))
    ppaa_cpu = numpy.zeros((nmo, nmo, ncas,ncas))
    papa_cpu = numpy.zeros((nmo, ncas, nmo,nmo))
    fxpp = numpy.empty ((nmo, nmo, naoaux))
    blksize = with_df.blockdim
    bufpa = numpy.empty((naoaux,nmo,ncas))
    fmmm = _ao2mo.libao2mo.AO2MOmmm_nr_s2_iltj
    fdrv = _ao2mo.libao2mo.AO2MOnr_e2_drv
    ftrans = _ao2mo.libao2mo.AO2MOtranse2_nr_s2
    bufs1 = numpy.empty((blksize,nmo,nmo))
    for k, eri1 in enumerate(with_df.loop(blksize)):
        naux = eri1.shape[0]
        bufpp = bufs1[:naux]
        fdrv(ftrans, fmmm,
         bufpp.ctypes.data_as(ctypes.c_void_p),
         eri1.ctypes.data_as(ctypes.c_void_p),
         mo.ctypes.data_as(ctypes.c_void_p),
         ctypes.c_int(naux), ctypes.c_int(nao),
         (ctypes.c_int*4)(0, nmo, 0, nmo),
         ctypes.c_void_p(0), ctypes.c_int(0))
        fxpp_keys.append([k, b0, b0+naux])
        fxpp[:,:,b0:b0+naux] = bufpp.transpose(1,2,0)  
        bufpa[b0:b0+naux] = bufpp[:,:,ncore:nocc]
        bufd = numpy.einsum('kii->ki', bufpp)
        j_pc_cpu += numpy.einsum('ki,kj->ij', bufd, bufd[:,:ncore])
        k_cp += numpy.einsum('kij,kij->ij', bufpp[:,:ncore], bufpp[:,:ncore])
        b0 += naux
    #bufs1 = bufpp = None
    k_pc_cpu = k_cp.T.copy()
    bufaa = bufpa[:,ncore:nocc,:].copy()#.reshape(-1,ncas**2)
    for p0, p1 in prange(0, nmo, nblk):
        nrow = p1 - p0
        buf = bufs1[:nrow]
        tmp = bufs2[:nrow].reshape(-1,ncas**2)
        for key, col0, col1 in fxpp_keys:
        #buf[:nrow,:,col0:col1] = fxpp[key][p0:p1]
            buf[:nrow,:,col0:col1] = fxpp[:,:,col0:col1][p0:p1]
        lib.dot(buf.reshape(-1,naoaux), bufaa, 1, tmp)
        ppaa_cpu[p0:p1] = tmp.reshape(p1-p0,nmo,ncas,ncas)
    bufs1 = numpy.empty((nblk,ncas,nmo,ncas))
    dgemm = lib.numpy_helper._dgemm
    for p0, p1 in prange(0, nmo, nblk):
        tmp = numpy.dot(bufpa[:,p0:p1].reshape(naoaux,-1).T,
                            bufpa.reshape(naoaux,-1))
        papa_cpu[p0:p1] = tmp.reshape(p1-p0,ncas,nmo,ncas)
 

    return j_pc_cpu, k_pc_cpu, ppaa_cpu, papa_cpu


def init_eri_gpu_v3 (mo, casscf, with_df):
    mo = numpy.asarray(mo, order='F')
    fxpp = numpy.empty ((nmo, nmo, naoaux))
    bufpa = numpy.empty((naoaux,nmo,ncas))
    b0 = 0
    j_pc = numpy.zeros((nmo,ncore))
    k_pc = numpy.zeros((nmo,ncore))
    k_cp = numpy.zeros((ncore,nmo))
    ppaa = numpy.zeros((nmo, nmo, ncas,ncas))
    papa = numpy.zeros((nmo, ncas, nmo,ncas))
    if gpu:
        arg = numpy.array([-1, -1, -1, -1], dtype=numpy.int32)
        libgpu.libgpu_get_dfobj_status(gpu, id(with_df), arg)
        if arg[2] > -1: load_eri = False
    libgpu.libgpu_push_mo_coeff(gpu, mo, nao*nmo)
    libgpu.libgpu_init_jk_ao2mo(gpu, ncore, nmo) # initializes j_pc and k_pc to be pulled
    libgpu.libgpu_init_ints_ao2mo_v3(gpu, naoaux, nmo, ncas) #initializes bufpa on pinned memory
    libgpu.libgpu_init_ppaa_ao2mo(gpu, nmo, ncas) #initializes ppaa on pinned memory
    count = 0
    for k, eri1 in enumerate(with_df.loop(blksize)):
        naux = eri1.shape[0]
        b0+=naux
    for count in range(k+1):
        arg = numpy.array([-1, -1, count, -1], dtype = numpy.int32)
        libgpu.libgpu_get_dfobj_status(gpu, id(with_df),arg)
        naux = arg[0]
        libgpu.libgpu_df_ao2mo_v3(gpu,blksize,nmo,nao,ncore,ncas,naux,eri1,count,id(with_df)) 
    libgpu.libgpu_pull_jk_ao2mo (gpu, j_pc, k_cp, nmo, ncore)
    libgpu.libgpu_pull_ints_ao2mo_v3(gpu, bufpa, blksize, naoaux, nmo, ncas)
    libgpu.libgpu_pull_ppaa_ao2mo(gpu, ppaa, nmo, ncas) #pull ppaa
    k_pc = k_cp.T.copy()

    for p0, p1 in prange(0, nmo, blksize):
        tmp = numpy.dot(bufpa[:,p0:p1].reshape(naoaux,-1).T,
                            bufpa.reshape(naoaux,-1))
        papa[p0:p1] = tmp.reshape(p1-p0,ncas,nmo,ncas)

    return j_pc,k_pc, ppaa, papa

def init_eri_gpu_v4 (mo, casscf, with_df):
    mo = numpy.asarray(mo, order='F')
    fxpp = numpy.empty ((nmo, nmo, naoaux))
    bufpa = numpy.empty((naoaux,nmo,ncas))
    b0 = 0
    j_pc = numpy.zeros((nmo,ncore))
    k_pc = numpy.zeros((nmo,ncore))
    k_cp = numpy.zeros((ncore,nmo))
    ppaa = numpy.zeros((nmo, nmo, ncas,ncas))
    papa = numpy.zeros((nmo, ncas, nmo,ncas))
    if gpu:
        arg = numpy.array([-1, -1, -1, -1], dtype=numpy.int32)
        libgpu.libgpu_get_dfobj_status(gpu, id(with_df), arg)
        if arg[2] > -1: load_eri = False
    libgpu.libgpu_push_mo_coeff(gpu, mo, nao*nmo)
    libgpu.libgpu_init_jk_ao2mo(gpu, ncore, nmo) 
    libgpu.libgpu_init_ppaa_papa_ao2mo(gpu, nmo, ncas) 
    count = 0
    for k, eri1 in enumerate(with_df.loop(blksize)):
        naux = eri1.shape[0]
        b0+=naux
    for count in range(k+1):
        arg = numpy.array([-1, -1, count, -1], dtype = numpy.int32)
        libgpu.libgpu_get_dfobj_status(gpu, id(with_df),arg)
        naux = arg[0]
        libgpu.libgpu_df_ao2mo_v4(gpu,blksize,nmo,nao,ncore,ncas,naux,eri1,count,id(with_df)) 
    libgpu.libgpu_pull_jk_ao2mo_v4 (gpu, j_pc, k_cp, nmo, ncore)
    libgpu.libgpu_pull_ppaa_papa_ao2mo_v4(gpu, ppaa, papa, nmo, ncas) #pull ppaa
    k_pc = k_cp.T.copy()

    return j_pc,k_pc, ppaa, papa

#Warm up iteration
for _ in range(1): j_pc_v3, k_pc_v3, ppaa_v3, papa_v3 = init_eri_gpu_v3 (mf.mo_coeff, mc, with_df) 
for _ in range(1): j_pc_v4, k_pc_v4, ppaa_v4, papa_v4 = init_eri_gpu_v4 (mf.mo_coeff, mc, with_df) 
if DEBUG: 
    print('i_pc check ',numpy.allclose(j_pc_v4,j_pc_v3))
    print('k_pc check ',numpy.allclose(k_pc_v4,k_pc_v3))
    print('ppaa check ',numpy.allclose(ppaa_v4,ppaa_v3))
    print('papa check ',numpy.allclose(papa_v4,papa_v3))

if PERFORMANCE:
    t0=time.time()
    for _ in range(nruns): 
        init_eri_gpu_v3 (mf.mo_coeff, mc, with_df) 
    t1=time.time()
    for _ in range(nruns): 
        init_eri_gpu_v4 (mf.mo_coeff, mc, with_df) 
    t2=time.time()
    print("Time v3", round(t1-t0,2))
    print("Time v4", round(t2-t1,2))
