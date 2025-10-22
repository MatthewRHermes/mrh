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

from tdm12s_performance import performance_checker as tdm12s_pc
from tppdm_performance import performance_checker as tppdm_pc
from sfudm1_performance import performance_checker as sfudm1_pc
from tdm13h_performance import performance_checker as tdm13h_pc

if __name__ == "__main__": 
  if gpu_run:
    gpu = libgpu.init()
    #libgpu.set_verbose_(gpu,1)
    from pyscf.lib import param
    param.use_gpu = gpu
    param.custom_fci=True
    param.mgpu_fci=True
    param.mgpu_fci_debug=True
  lib.logger.TIMER_LEVEL=lib.logger.INFO
  geom = ''' K 0 0 0;
           K 0 0 2;
           K 0 0 4;
           K 0 0 8;
           K 0 0 10;
           K 0 0 12;'''
  basis = 'def2tzvp'
  if gpu_run: mol = gto.M(use_gpu = gpu, atom=geom, basis=basis, verbose=1)
  else: mol = gto.M(atom=geom, basis=basis, verbose=1)

  mol.output='test.log'
  mol.build()
  
  mf = scf.RHF(mol)
  mf=mf.density_fit()
  mf.with_df.auxbasis = pyscf.df.make_auxbasis(mol)
  mf.max_cycle=1
  mf.kernel()
 
  norb = 11
  nelec = 15
  n_bra = 5
  n_ket = 5
  nruns = 6
  
  #TDM12s performance runs
  gpu_time, cpu_time = tdm12s_pc(n_bra, n_ket, norb, nelec, nruns = nruns)
  print("For tdm12s with" +" GPU time: ", round(gpu_time,2), "CPU time: ", round(cpu_time,2))
  print("Speedup = ", round(cpu_time/gpu_time, 2))
  param.use_gpu = gpu
  
  #TPPDM performance runs
  for spin in range(3):
    gpu_time, cpu_time = tppdm_pc(n_bra, n_ket, norb, nelec, spin, nruns = nruns)
    print("For tppdm GPU time: ", round(gpu_time,2), "CPU time: ", round(cpu_time,2))
    print("Speedup = ", round(cpu_time/gpu_time, 2))
    param.use_gpu = gpu
  
  #SFUDM performance runs
  gpu_time, cpu_time = sfudm1_pc( n_bra, n_ket, norb, nelec, nruns = nruns)
  print("For sfudm GPU time: ", round(gpu_time,2), "CPU time: ", round(cpu_time,2))
  print("Speedup = ", round(cpu_time/gpu_time, 2))
  param.use_gpu = gpu
  
  for spin, reorder in itertools.product(range(2), range(2)): 
    gpu_time, cpu_time = tdm13h_pc(0, n_bra, n_ket, norb, nelec, spin, reorder, nruns = nruns)
    print("For tdm13h GPU time: ", round(gpu_time,2), "CPU time: ", round(cpu_time,2))
    print("Speedup = ", round(cpu_time/gpu_time, 2))
    param.use_gpu = gpu
  libgpu.destroy_device(gpu)
