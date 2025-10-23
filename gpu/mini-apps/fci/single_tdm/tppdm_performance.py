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
from mrh.my_pyscf.fci.rdm import trans_ppdm
import time


def performance_checker(norb, nelec, spin=0, nruns = 10):
  s1 = int (spin>1)
  s2 = int (spin>0)
  ndum = 2 - (spin%2)
  nelec_ket = _unpack_nelec (nelec)
  nelec_bra = list (_unpack_nelec (nelec))
  nelec_bra[s1] += 1
  nelec_bra[s2] += 1
  occ_a, occ_b = int (spin<2), int (spin>0)
  na_bra = math.comb(norb, nelec_bra[0])
  nb_bra = math.comb(norb, nelec_bra[1])
  na_ket = math.comb(norb, nelec_ket[0])
  nb_ket = math.comb(norb, nelec_ket[1])
  cibra = np.random.random((na_bra, nb_bra))
  ciket = np.random.random((na_ket, nb_ket))

  t0 = time.time()
  for _ in range(nruns):
    trans_ppdm (cibra, ciket, norb, nelec, spin = spin) 
  t1 = time.time()
  param.use_gpu = None
  for _ in range(nruns): 
    trans_ppdm (cibra, ciket, norb, nelec, spin = spin) 
  t2 = time.time()
  return t1-t0, t2-t1

if __name__ == "__main__": 
  if gpu_run:
    gpu = libgpu.init()
    libgpu.set_verbose_(gpu,1)
    from pyscf.lib import param
    param.use_gpu = gpu
    #param.gpu_debug=True
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
  
  for spin in range(2): 
      gpu_time, cpu_time = performance_checker(norb, nelec, spin)
      print("GPU time: ", round(gpu_time,2), "CPU time: ", round(cpu_time,2))
  libgpu.destroy_device(gpu)
