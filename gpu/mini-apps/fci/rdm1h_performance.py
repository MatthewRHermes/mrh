gpu_run=1
if gpu_run:from mrh.my_pyscf.gpu import libgpu
import pyscf
import numpy as np 
if gpu_run:from gpu4mrh import patch_pyscf
from pyscf import gto, scf, tools, mcscf, lib
from pyscf.fci import  cistring
from mrh.my_pyscf.fci.rdm import _trans_rdm1hs
from pyscf.fci.addons import _unpack_nelec
from mrh.my_pyscf.mcscf.lasscf_async import LASSCF
import math
import time
import itertools

if gpu_run:
  gpu = libgpu.init()
  from pyscf.lib import param
  param.use_gpu = gpu
  libgpu.set_verbose_(gpu, 1)
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

def performance_checker(cre, norb, nelec, spin=0, link_index = None, nruns = 10):
  param.use_gpu = gpu
  nelec_copy = list(_unpack_nelec(nelec))
  print(cre, nelec, spin)
  if not cre:
      nelec_copy[spin] -=1
  nelec_ket = _unpack_nelec(nelec_copy)
  nelec_bra = [x for x in nelec_copy]
  nelec_bra[spin] += 1
  
  na_bra = math.comb(norb, nelec_bra[0])
  nb_bra = math.comb(norb, nelec_bra[1])
  na_ket = math.comb(norb, nelec_ket[0])
  nb_ket = math.comb(norb, nelec_ket[1])
  cibra = np.random.random((na_bra, nb_bra))
  ciket = np.random.random((na_ket, nb_ket))
  if not cre: 
    cibra, ciket = ciket, cibra

  t0 = time.time()
  for _ in range(nruns):
    _trans_rdm1hs(cre, cibra, ciket, norb, nelec, spin=spin, link_index = link_index)
  t1 = time.time()
  param.use_gpu = None
  for _ in range(nruns): 
    _trans_rdm1hs(cre, cibra, ciket, norb, nelec, spin=spin, link_index = link_index)
  t2 = time.time()
  return t1-t0, t2-t1


norb = 12
nelec = 15

for cre, spin in itertools.product(range(2), range(2)): 
    print(cre, spin)
    gpu_time, cpu_time = performance_checker(cre, norb, nelec, spin, link_index= None)
    print("GPU time: ", round(gpu_time,2), "CPU time: ", round(cpu_time,2))
libgpu.destroy_device(gpu)
