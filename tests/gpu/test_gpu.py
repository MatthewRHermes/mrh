import unittest
from pyscf import gto, scf, tools, mcscf,lib
from mrh.my_pyscf.mcscf.lasscf_async import LASSCF
from pyscf.mcscf import avas
from mrh.tests.gpu.geometry_generator import generator

def setUpModule():
    global nfrags, basis
    nfrags = 4
    basis = '631g'
    
def tearDownModule():
    global nfrags, basis
    del nfrags, basis

def _run_mod (gpu_run):
    if gpu_run: 
        from mrh.my_pyscf.gpu import libgpu
        from gpu4mrh import patch_pyscf
        gpu = libgpu.libgpu_init()
        outputfile=str(nfrags)+'_'+str(basis)+'_out_gpu_ref.log';
        mol=gto.M(atom=generator(nfrags),basis=basis,verbose=4,output=outputfile, use_gpu=gpu)
    else: 
        outputfile=str(nfrags)+'_'+str(basis)+'_out_cpu_ref.log';
        mol=gto.M(atom=generator(nfrags),basis=basis,verbose=4,output=outputfile)
    mf=scf.RHF(mol)
    mf=mf.density_fit()
    mf.run()
    if gpu_run: 
        las=LASSCF(mf, list((2,)*nfrags),list((2,)*nfrags),verbose=4,use_gpu=gpu)
    else:
        las=LASSCF(mf, list((2,)*nfrags),list((2,)*nfrags),verbose=4)
    frag_atom_list=[list(range(1+4*nfrag,3+4*nfrag)) for nfrag in range(nfrags)]
    ncas,nelecas,guess_mo_coeff=avas.kernel(mf, ["C 2pz"])
    mo_coeff=las.set_fragments_(frag_atom_list, guess_mo_coeff)
    las.kernel(mo_coeff)
    if gpu_run: libgpu.libgpu_destroy_device(gpu)
    return mf, las

class KnownValues (unittest.TestCase):

    def test_implementations (self):
        mf_gpu, las_gpu = _run_mod (gpu_run=True)
        with self.subTest ('GPU accelerated calculation converged'):
            self.assertTrue (mf_gpu.converged)
            self.assertTrue (las_gpu.converged)
        mf_cpu, las_cpu = _run_mod (gpu_run=False)
        with self.subTest ('CPU-only calculation converged'):
            self.assertTrue (mf_cpu.converged)
            self.assertTrue (las_cpu.converged)
        with self.subTest ('Total energy'):
            self.assertAlmostEqual (las_cpu.e_tot, las_gpu.e_tot, 7)
            self.assertAlmostEqual (mf_cpu.e_tot, mf_gpu.e_tot, 7)

if __name__ == "__main__":
    print("Tests for GPU accelerated LASSCF")
    unittest.main()
