''' 
This miniapp is for development of gpu accelerated version of creation of impurity subspace cholesky vectors by the following equations

     
     b^P_{e_{K_1}\mu_2} = b^P_{\mu_1\mu_2}M^{\mu_1}_{e_{K_1}} 
     b^P_{e_{K_1}e_{K_2}} = b^P_{e_{K_1}\mu_2}M^{\mu_2}_{e_{K_2}} 

'''
import unittest
from pyscf import gto, scf, tools, mcscf,lib, ao2mo
from mrh.my_pyscf.mcscf.lasscf_async import LASSCF
from pyscf.mcscf import avas
from mrh.tests.gpu.geometry_generator import generator
import numpy as np

def setUpModule():
    global nfrags, basis, nimp
    nfrags = 4
    basis = '631g'
    nimp = 10
    
    
def tearDownModule():
    global nfrags, basis, nimp
    del nfrags, basis, nimp

def _run_mod (imporb_coeff, return_4c2eeri=True, gpu_run=True):

    from pyscf.lib import param
    if gpu_run: 
        from mrh.my_pyscf.gpu import libgpu
        from gpu4mrh import patch_pyscf
        gpu = libgpu.init()
        outputfile=str(nfrags)+'_'+str(basis)+'_out_gpu_ref.log';
        mol=gto.M(atom=generator(nfrags),basis=basis,verbose=4,output=outputfile, use_gpu=gpu)
        param.use_gpu = gpu
    else: 
        param.use_gpu = None
        outputfile=str(nfrags)+'_'+str(basis)+'_out_cpu_ref.log';
        mol=gto.M(atom=generator(nfrags),basis=basis,verbose=4,output=outputfile)
    mf=scf.RHF(mol)
    mf=mf.density_fit()
    mf.run()

    if gpu_run:
         def impham_gpu_v1(imporb_coeff,return_4c2eeri):
            nao_s,nao_f = imporb_coeff.shape
            naoaux = mf.with_df.get_naoaux()
            blksize=mf.with_df.blockdim
            #_cderi=np.empty((naoaux, nao_f, nao_f),dtype=imporb_coeff.dtype)
            if return_4c2eeri:   
                cderi=np.zeros((nao_f*(nao_f+1)//2, nao_f*(nao_f+1)//2),dtype=imporb_coeff.dtype)
            else: 
                cderi=np.zeros((mf.with_df.get_naoaux(), nao_f*(nao_f+1)//2),dtype=imporb_coeff.dtype)
            libgpu.push_mo_coeff(gpu, imporb_coeff, nao_s*nao_f)
            libgpu.init_eri_impham(gpu, naoaux, nao_f,return_4c2eeri)
            for k, eri1 in enumerate(mf.with_df.loop(blksize)):pass;
            for count in range(k+1): 
                arg = np.array([-1, -1, count, -1], dtype = np.int32)
                libgpu.get_dfobj_status(gpu, id(mf.with_df),arg)
                naux = arg[0]
                libgpu.compute_eri_impham (gpu, nao_s, nao_f, blksize, naux, count, id(mf.with_df),return_4c2eeri)
            libgpu.pull_eri_impham(gpu, cderi, naoaux, nao_f,return_4c2eeri)  
            return cderi
         return impham_gpu_v1(imporb_coeff, return_4c2eeri) 
                          
    else:
        def impham_cpu_original(imporb_coeff, return_4c2eeri):
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
            if return_4c2eeri:
                return np.dot (_cderi.conj ().T, _cderi)
            else: 
                return _cderi
        return impham_cpu_original(imporb_coeff, return_4c2eeri) 


class KnownValues (unittest.TestCase):

    def test_implementations (self):
        mol=gto.M(atom=generator(nfrags),basis=basis)
        imporb_coeff=np.random.random((mol.nao, nimp))
        eri_cpu = _run_mod (imporb_coeff, return_4c2eeri=True, gpu_run=False)
        eri_gpu = _run_mod (imporb_coeff, return_4c2eeri=True, gpu_run=True)
        eri_diff = np.max(np.abs(eri_cpu - eri_gpu))
        with self.subTest ('GPU accelerated eri creation'):
            self.assertAlmostEqual (eri_diff, 0, 7)

if __name__ == "__main__":
    print("Tests for GPU accelerated embedding ERI/CDERI creation")
    unittest.main()

 
