gpu_run=1
if gpu_run:from mrh.my_pyscf.gpu import libgpu
import pyscf
import numpy as np 
if gpu_run:from gpu4mrh import patch_pyscf
from pyscf import gto, scf, tools, mcscf, lib
from pyscf.lib import param
from pyscf.fci import  cistring
from pyscf.fci.addons import _unpack_nelec
from mrh.my_pyscf.mcscf.lasscf_async import LASSCF
import math
import time
import itertools

from rdm13h_performance import performance_checker as rdm13h_pc
#def performance_checker(cre, norb, nelec, spin=0, link_index = None, reorder=False, nruns = 10):
from sfudm1_performance import performance_checker as sfudm_pc
#def performance_checker(norb, nelec, nruns = 10):
from tppdm_performance import performance_checker as tppdm_pc
#def performance_checker(norb, nelec, spin=0, nruns = 10):
from rdm2_performance import performance_checker as rdm2_pc
#def performance_checker(fn, norb, nelec, nruns=10):


if __name__ == "__main__": 
  if gpu_run:
    gpu = libgpu.init()
    from pyscf.lib import param
    param.use_gpu = gpu
    libgpu.set_verbose_(gpu, 1)
    param.custom_fci=True
  lib.logger.TIMER_LEVEL=lib.logger.INFO
  
  geom = ''' K 0 0 0;
             K 0 0 2;'''
  
  if gpu_run: mol = gto.M(use_gpu = gpu, atom=geom, basis='631g', verbose=1)
  else: mol = gto.M(atom=geom, basis='631g', verbose=1)
  
  mol.output='test.log'
  mol.build()
  
  mf = scf.RHF(mol)
  mf=mf.density_fit()
  mf.with_df.auxbasis = pyscf.df.make_auxbasis(mol)
  mf.max_cycle=1
  mf.kernel()
  
  
  norb = 11
  nelec = 15
  
  for fn in ['FCItdm12kern_a', 'FCItdm12kern_b', 'FCItdm12kern_ab', 'FCIrdm12kern_sf']: 
    gpu_time, cpu_time = rdm2_pc(fn, norb, nelec)
    print("For RDM2 with" + fn +" GPU time: ", round(gpu_time,2), "CPU time: ", round(cpu_time,2))
    param.use_gpu = gpu
  for spin in range(2): 
    gpu_time, cpu_time = tppdm_pc(norb, nelec, spin)
    print("For tppdm GPU time: ", round(gpu_time,2), "CPU time: ", round(cpu_time,2))
    param.use_gpu = gpu
  
  gpu_time, cpu_time = sfudm_pc( norb, nelec, nruns = 20)
  print("GPU time: ", round(gpu_time,2), "CPU time: ", round(cpu_time,2))
  param.use_gpu = gpu
  for cre, spin, reorder in itertools.product(range(2), range(2), range(2)): 
    gpu_time, cpu_time = rdm13h_pc(cre, norb, nelec, spin, link_index= None, reorder=reorder)
    print("For rdm13h GPU time: ", round(gpu_time,2), "CPU time: ", round(cpu_time,2))
    param.use_gpu = gpu
  libgpu.destroy_device(gpu)
