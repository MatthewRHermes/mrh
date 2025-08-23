gpu_run=1
if gpu_run:from mrh.my_pyscf.gpu import libgpu
import pyscf
import numpy as np 
if gpu_run:from gpu4mrh import patch_pyscf
from pyscf import gto, scf, tools, mcscf, lib
from pyscf.fci import  cistring
from mrh.my_pyscf.fci.rdm import _trans_rdm13hs
from pyscf.fci.addons import _unpack_nelec
from mrh.my_pyscf.mcscf.lasscf_async import LASSCF
import math
import time

if gpu_run:
  gpu = libgpu.init()
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

def performance_checker(cibra, ciket, norb, nelec, spin=0, link_index = None, reorder=False, nruns = 6):
  param.use_gpu = gpu
  t0 = time.time()
  for _ in range(nruns):
    _trans_rdm13hs(False, cibra, ciket, norb, nelec, spin=0, link_index = link_index, reorder=True)
  t1 = time.time()
  param.use_gpu = None
  for _ in range(nruns): 
    _trans_rdm13hs(False, cibra, ciket, norb, nelec, spin=0, link_index = link_index, reorder=True)
  t2 = time.time()
  print("GPU time: ", round(t1-t0,2), "CPU time: ", round(t2-t1,2))


norb = 10
nelec = 9
neleca, nelecb = _unpack_nelec(nelec)
na = math.comb(norb, neleca-1)
nb = math.comb(norb, nelecb)
cibra = np.random.random((na,nb))
na = math.comb(norb, neleca)
ciket = np.random.random((na,nb))
link_indexa = cistring.gen_linkstr_index(range(norb+1), neleca)
link_indexb = cistring.gen_linkstr_index(range(norb+1), nelecb)
link_index = (link_indexa, link_indexb)
#_trans_rdm13hs(True, cibra, ciket, norb, nelec, spin=0, link_index = link_index, reorder=True)
performance_checker(cibra, ciket, norb, nelec, spin=0, link_index= link_index, reorder=False)
