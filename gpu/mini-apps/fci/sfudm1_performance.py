gpu_run=1
if gpu_run:from mrh.my_pyscf.gpu import libgpu
import pyscf
import numpy as np 
if gpu_run:from gpu4mrh import patch_pyscf
from pyscf import gto, scf, lib
from pyscf.lib import param
from pyscf.fci import  cistring
from pyscf.fci.addons import _unpack_nelec
import math
from mrh.my_pyscf.fci.rdm import trans_sfudm1 
import time
def performance_checker(norb, nelec, nruns = 10):
  nelec_ket = _unpack_nelec(nelec)
  nelec_bra = list(_unpack_nelec(nelec))
  nelec_bra[0] +=1
  nelec_bra[1] -=1
  na_bra = math.comb(norb, nelec_bra[0])
  nb_bra = math.comb(norb, nelec_bra[1])
  na_ket = math.comb(norb, nelec_ket[0])
  nb_ket = math.comb(norb, nelec_ket[1])
  cibra = np.random.random((na_bra, nb_bra))
  ciket = np.random.random((na_ket, nb_ket))

  t0 = time.time()
  for _ in range(nruns):
    trans_sfudm1 (cibra, ciket, norb, nelec) 
  t1 = time.time()
  param.use_gpu = None
  for _ in range(nruns): 
    trans_sfudm1 (cibra, ciket, norb, nelec) 
  t2 = time.time()
  return t1-t0, t2-t1

if __name__ == "__main__": 
  if gpu_run:
    gpu = libgpu.init()
    from pyscf.lib import param
    param.use_gpu = gpu
    #param.gpu_debug=True
    param.custom_fci=True
    libgpu.set_verbose_(gpu, 1)
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
  
  gpu_time, cpu_time = performance_checker( norb, nelec)
  print("GPU time: ", round(gpu_time,2), "CPU time: ", round(cpu_time,2))
  libgpu.destroy_device(gpu)
