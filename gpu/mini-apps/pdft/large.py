gpu_run=1
if gpu_run:from mrh.my_pyscf.gpu import libgpu
import pyscf
import numpy as np 
if gpu_run:from gpu4mrh import patch_pyscf
#from mrh.tests.gpu.geometry_generator import generator
from pyscf import gto, scf, tools, mcscf, lib, mcpdft
from mrh.my_pyscf.mcscf.lasscf_async import LASSCF
from pyscf.mcscf import avas	
if gpu_run:gpu = libgpu.init()
lib.logger.TIMER_LEVEL=lib.logger.INFO

#if gpu_run: libgpu.set_verbose_(gpu, 1)
geom = '''Kr 0 0 0; 
                  Kr 0 0 10; 
                  Kr 0 0 20; 
                  Kr 0 0 30; 
                  Kr 0 0 40; 
                  Kr 0 0 50; 
                  Kr 0 0 60; 
                  Kr 0 0 70; 
                  Kr 0 0 80; 
                  Kr 0 0 90; 
                  Kr 0 0 100; 
                  Kr 0 0 110; 
                  Kr 0 0 120;'''
for i in range(1, 21, 2):
    ncas = i*2
    if gpu_run: mol = gto.M(use_gpu=gpu,atom=geom, basis = 'sto3g',spin = 0, verbose=4)#
    else: mol = gto.M(atom=geom, basis = 'sto3g',spin = 0, verbose=4)#
    mol.output=f'mcpdft.{ncas}.out'
    if gpu_run: mol.output=f'gpu_mcpdft.{ncas}.out'
    mol.build()
    mf = scf.RHF (mol)
    mf=mf.density_fit()
    mf.max_cycle=1
    mf.kernel()
    nelecas = 2*ncas
    mc0 = mcpdft.CASCI (mf, 'tPBE', ncas, nelecas, grids_level=9)
    mc0.fcisolver.make_rdm1s = lambda *arg: [np.random.rand(ncas, ncas),]*2
    mc0.fcisolver.make_rdm2 = lambda *arg: np.random.rand(ncas, ncas, ncas, ncas)
    mc0.compute_pdft_energy_(dump_chk=False)
