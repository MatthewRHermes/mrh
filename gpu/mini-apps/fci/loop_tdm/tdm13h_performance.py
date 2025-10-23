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
import time
import tdm13h_test
from itertools import product

def performance_checker(cre, n_bra, n_ket, norb, nelec, spin, reorder, nruns=10):
    nelec_copy = list(_unpack_nelec(nelec))
    cre = False
    if not cre:
        nelec_copy[spin] -=1
    nelec_ket = _unpack_nelec(nelec_copy)
    nelec_bra = [x for x in nelec_copy]
    nelec_bra[spin] += 1

    na_bra = math.comb(norb, nelec_bra[0])
    nb_bra = math.comb(norb, nelec_bra[1])
    na_ket = math.comb(norb, nelec_ket[0])
    nb_ket = math.comb(norb, nelec_ket[1])

    bravecs = np.empty((n_bra, na_bra, nb_bra))
    ketvecs = np.empty((n_ket, na_ket, nb_ket))
    for _nbra in range(n_bra): bravecs[_nbra] = np.random.random((na_bra, nb_bra))
    for _nket in range(n_ket): ketvecs[_nket] = np.random.random((na_ket, nb_ket))
 
    from pyscf.lib import param
    try: 
      use_gpu = param.use_gpu
      gpu = use_gpu
    except: 
      use_gpu = None
    try: gpu_debug = param.gpu_debug
    except: gpu_debug = False
    try: custom_fci = param.custom_fci
    except: custom_fci = False
    try: custom_debug = param.custom_debug
    except: custom_debug = False
    try: mgpu_fci = param.mgpu_fci
    except: mgpu_fci = False
    try: mgpu_fci_debug = param.mgpu_fci_debug
    except: mgpu_fci_debug = False

    print(gpu)
    t0 = time.time()
    for _ in range(nruns):
      tdm1h, tdm3h = tdm13h_test.multi_gpu_loop(cre, bravecs, ketvecs,norb,nelec, spin, reorder)
    t1 = time.time()
    param.use_gpu = None
    for _ in range(nruns): 
      tdm1h, tdm3h = tdm13h_test.o0_loop(cre, bravecs, ketvecs,norb,nelec, spin, reorder)
    t2 = time.time()
    param.use_gpu = gpu
    return t1-t0, t2-t1


    


if __name__ == "__main__": 
  if gpu_run:
    gpu = libgpu.init()
    libgpu.set_verbose_(gpu,1)
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
  n_bra, n_ket = 5,6
  nruns = 2
  
  for cre, spin, reorder in product( range(2), range(2), range(2)): 
      gpu_time, cpu_time = performance_checker(cre, n_bra, n_ket, norb, nelec, spin, reorder, nruns = nruns)
      print("GPU time: ", round(gpu_time,2), "CPU time: ", round(cpu_time,2))
  libgpu.destroy_device(gpu)
