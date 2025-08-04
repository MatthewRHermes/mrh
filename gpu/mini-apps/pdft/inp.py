gpu_run=1
N=0
if gpu_run:from mrh.my_pyscf.gpu import libgpu
import pyscf
import numpy as np 
if gpu_run:from gpu4mrh import patch_pyscf
from gpu4mrh import patch_pyscf
from mrh.tests.gpu.geometry_generator import generator
from pyscf import gto, scf, tools, mcscf, lib, mcpdft
from mrh.my_pyscf.mcscf.lasscf_async import LASSCF
from pyscf.mcscf import avas	
if gpu_run:gpu = libgpu.init()
lib.logger.TIMER_LEVEL=lib.logger.INFO
nfrags=3;basis='631g';
#if N:
#    atom='''Li 0.0 0.0 0.0;
#    Li 0.0 0.0 1.0'''
#    output_file = "test_file.log"
#    mol=gto.M(atom=atom,basis=basis,verbose=5)
#    if gpu_run:mol=gto.M(use_gpu=gpu, atom=atom,basis=basis,verbose=5, output=output_file)
#else:
outputfile=str(nfrags)+'_'+str(basis)+'_cpu.log';
if gpu_run: outputfile=str(nfrags)+'_'+str(basis)+'_gpu_new_2.log';
mol=gto.M(atom=generator(nfrags),basis=basis,verbose=4,output=outputfile,max_memory=160000)
if gpu_run:mol=gto.M(use_gpu=gpu, atom=generator(nfrags),basis=basis,verbose=4,output=outputfile,max_memory=160000)
mf=scf.RHF(mol)
mf=mf.density_fit(auxbasis='weigend')
mf.run()
#if N:
#    las=LASSCF(mf, (1,1),(1,1),verbose=4)
#    if gpu_run: las=LASSCF(mf, (1,1),(1,1),verbose=4,use_gpu=gpu,output=output_file)
#    las.set_fragments_([(0,),(1,)],mf.mo_coeff)
#    mo_coeff=mf.mo_coeff
#else:
    #las=LASSCF(mf, list((2,)*nfrags),list((2,)*nfrags),verbose=4)#, use_gpu=gpu)
    #if gpu_run:las=LASSCF(mf, list((2,)*nfrags),list((2,)*nfrags), use_gpu=gpu,verbose=4)
    #frag_atom_list=[list(range(1+4*nfrag,3+4*nfrag)) for nfrag in range(nfrags)]
    #ncas,nelecas,guess_mo_coeff=avas.kernel(mf, ["C 2pz"])
    #mo_coeff=las.set_fragments_(frag_atom_list, guess_mo_coeff)
#las.max_cycle_macro=4
#if gpu_run: las = las(use_gpu=gpu)
#las.kernel(mo_coeff)
#print(las.e_tot)
ncas,nelecas,guess_mo_coeff=avas.kernel(mf, ["C 2pz"])
mc0 = mcpdft.CASCI (mf, 'tPBE', ncas, nelecas, verbose=4, grids_level=2)
#mc0.fcisolver.make_rdm1s = lambda *arg: [np.random.rand(ncas, ncas),]*2
#mc0.fcisolver.make_rdm2 = lambda *arg: np.random.rand(ncas, ncas, ncas, ncas)
mc0.fcisolver.make_rdm1s = lambda *arg: [np.arange(ncas*ncas).reshape(ncas, ncas)/(ncas*ncas),]*2
mc0.fcisolver.make_rdm2 = lambda *arg: np.arange(ncas**4).reshape(ncas, ncas, ncas, ncas)/(ncas**4)
mc0.compute_pdft_energy_(dump_chk=False)
#print(mc0.analyze())
