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
import unittest
from pyscf import gto, scf, tools, mcscf,lib
from mrh.my_pyscf.mcscf.lasscf_async import LASSCF
from pyscf.mcscf import avas
from mrh.tests.gpu.geometry_generator import generator
from pyscf.ao2mo import _ao2mo
import ctypes
import numpy 

def setUpModule():
    global nfrags, basis
    nfrags = 2
    basis = 'sto3g'
    
def tearDownModule():
    global nfrags, basis
    del nfrags, basis

def _run_mod ():

    from mrh.my_pyscf.gpu import libgpu
    from gpu4mrh import patch_pyscf
    gpu = libgpu.init()
    outputfile=str(nfrags)+'_'+str(basis)+'_out_gpu_ref.log';
    mol=gto.M(atom=generator(nfrags),basis=basis,verbose=4,output=outputfile, use_gpu=gpu)
    mf=scf.RHF(mol)
    mf=mf.density_fit()
    mf.run()
    mc=mcscf.CASSCF(mf,2*nfrags, 2*nfrags)
    mo=mf.mo_coeff
    with_df = mc._scf.with_df
    mo = numpy.asarray(mo, order='F')
    nao,nmo=mo.shape
    ncore=mc.ncore
    ncas=mc.ncas
    naoaux = with_df.get_naoaux()
    nocc=ncas+ncore
    blksize=with_df.blockdim
    #j_pc = numpy.zeros((nmo,ncore))
    #k_pc = numpy.zeros((nmo,ncore))
    #k_cp = numpy.zeros((ncore,nmo))
    #ppaa = numpy.zeros((nmo, nmo, ncas,ncas))
    #papa = numpy.zeros((nmo, ncas, nmo,ncas))
    def init_eri_cpu (mo, with_df):
        fxpp = numpy.empty ((nmo, nmo, naoaux))
        bufpa = numpy.empty((naoaux,nmo,ncas))
        fxpp_keys = []
        b0 = 0
        j_pc = numpy.zeros((nmo,ncore))
        k_pc = numpy.zeros((nmo,ncore))
        k_cp = numpy.zeros((ncore,nmo))
        ppaa = numpy.zeros((nmo, nmo, ncas,ncas))
        papa = numpy.zeros((nmo, ncas, nmo,ncas))
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
            j_pc += numpy.einsum('ki,kj->ij', bufd, bufd[:,:ncore])
            k_cp += numpy.einsum('kij,kij->ij', bufpp[:,:ncore], bufpp[:,:ncore])
            b0 += naux
        #bufs1 = bufpp = None
        k_pc = k_cp.T.copy()
        bufaa = bufpa[:,ncore:nocc,:].copy().reshape(-1,ncas**2)
        bufs1 = numpy.empty((blksize,nmo,naoaux))
        bufs2 = numpy.empty((blksize,nmo,ncas,ncas))
        for p0, p1 in prange(0, nmo, blksize):
            nrow = p1 - p0
            buf = bufs1[:nrow]
            tmp = bufs2[:nrow].reshape(-1,ncas**2)
            for key, col0, col1 in fxpp_keys:
            #buf[:nrow,:,col0:col1] = fxpp[key][p0:p1]
                buf[:nrow,:,col0:col1] = fxpp[:,:,col0:col1][p0:p1]
            lib.dot(buf.reshape(-1,naoaux), bufaa, 1, tmp)
            ppaa[p0:p1] = tmp.reshape(p1-p0,nmo,ncas,ncas)
        bufs1 = None
        bufs1 = numpy.empty((blksize,ncas,nmo,ncas))
        dgemm = lib.numpy_helper._dgemm
        for p0, p1 in prange(0, nmo, blksize):
            tmp = numpy.dot(bufpa[:,p0:p1].reshape(naoaux,-1).T,
                                bufpa.reshape(naoaux,-1))
            papa[p0:p1] = tmp.reshape(p1-p0,ncas,nmo,ncas)
     
    
        return j_pc, k_pc, ppaa, papa

    def init_eri_gpu_v4 (mo, with_df):
        b0 = 0
        j_pc = numpy.zeros((nmo,ncore))
        k_pc = numpy.zeros((nmo,ncore))
        k_cp = numpy.zeros((ncore,nmo))
        ppaa = numpy.zeros((nmo, nmo, ncas,ncas))
        papa = numpy.zeros((nmo, ncas, nmo,ncas))
        if gpu:
            arg = numpy.array([-1, -1, -1, -1], dtype=numpy.int32)
            libgpu.get_dfobj_status(gpu, id(with_df), arg)
            if arg[2] > -1: load_eri = False
        libgpu.push_mo_coeff(gpu, mo, nao*nmo)
        libgpu.init_jk_ao2mo(gpu, ncore, nmo) 
        libgpu.init_ppaa_papa_ao2mo(gpu, nmo, ncas) 
        count = 0
        for k, eri1 in enumerate(with_df.loop(blksize)):
            naux = eri1.shape[0]
            b0+=naux
        for count in range(k+1):
            arg = numpy.array([-1, -1, count, -1], dtype = numpy.int32)
            libgpu.get_dfobj_status(gpu, id(with_df),arg)
            naux = arg[0]
            libgpu.df_ao2mo_v4(gpu,blksize,nmo,nao,ncore,ncas,naux,count,id(with_df)) 
        libgpu.pull_jk_ao2mo_v4 (gpu, j_pc, k_cp, nmo, ncore)
        libgpu.pull_ppaa_papa_ao2mo_v4(gpu, ppaa, papa, nmo, ncas) #pull ppaa
        k_pc = k_cp.T.copy()
    
        return j_pc,k_pc, ppaa, papa
     

    def prange(start, end, step):
        for i in range(start, end, step):
            yield i, min(i+step, end)
    
    j_pc_gpu, k_pc_gpu, ppaa_gpu, papa_gpu = init_eri_gpu_v4(mf.mo_coeff,with_df) 
    j_pc_cpu, k_pc_cpu, ppaa_cpu, papa_cpu = init_eri_cpu(mf.mo_coeff,with_df) 
    j_pc_diff=numpy.max(numpy.abs(j_pc_cpu-j_pc_gpu))
    k_pc_diff=numpy.max(numpy.abs(k_pc_cpu-k_pc_gpu))
    ppaa_diff=numpy.max(numpy.abs(ppaa_cpu-ppaa_gpu))
    papa_diff=numpy.max(numpy.abs(papa_cpu-papa_gpu))
    return j_pc_diff, k_pc_diff, ppaa_diff, papa_diff
    
   

class KnownValues (unittest.TestCase):

    def test_implementations (self):
    ## VA 5/7/25 - I can't do a _run_mod(gpu_run=False) and _run_mod(gpu_run=True) and compare j_pc, k_pc, ppaa and papa in different runs
    ## because the ERIs can differ between runs in terms of sign (phase?), leading to answers differing by sign
        j_pc_diff, k_pc_diff, ppaa_diff, papa_diff=_run_mod()
        with self.subTest ('JKs'):
            self.assertAlmostEqual (j_pc_diff, 0, 7)
            self.assertAlmostEqual (k_pc_diff, 0, 7)
        with self.subTest ('Intermediates'):
            self.assertAlmostEqual (ppaa_diff, 0, 7)
            self.assertAlmostEqual (papa_diff, 0, 7)

if __name__ == "__main__":
    print("Tests for GPU accelerated AO2MO kernels")
    unittest.main()
