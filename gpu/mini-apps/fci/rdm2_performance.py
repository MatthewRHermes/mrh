gpu_run=1
if gpu_run:from mrh.my_pyscf.gpu import libgpu
import pyscf
import numpy as np 
if gpu_run:from gpu4mrh import patch_pyscf
from pyscf import gto, scf, tools, mcscf, lib
from pyscf.fci import rdm, cistring
from pyscf.fci.addons import _unpack_nelec
from mrh.my_pyscf.mcscf.lasscf_async import LASSCF
import time

if gpu_run:
  gpu = libgpu.init()
  from pyscf.lib import param
  param.use_gpu = gpu
  param.gpu_debug=False
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

norb = 12
nelec = 15

neleca, nelecb = _unpack_nelec(nelec)
link_indexa = link_indexb = cistring.gen_linkstr_index(range(norb), neleca)
if neleca != nelecb: link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)

na = link_indexa.shape[0]
nb = link_indexb.shape[0]
cibra = np.random.random((na,nb))
ciket = np.random.random((na,nb))
link_index = (link_indexa, link_indexb)

def performance_checker(fn, cibra, ciket, norb, nelec, link_index, nruns=5):
  
  param.use_gpu = gpu
  t0 = time.time()
  for _ in range(nruns): rdm.make_rdm12_spin1(fn, cibra, ciket, norb, nelec, link_index)
  t1 = time.time()
  param.use_gpu = None
  for _ in range(nruns): rdm.make_rdm12_spin1(fn, cibra, ciket, norb, nelec, link_index)
  t2 = time.time()
  print("GPU time: ", round(t1-t0,2), "CPU time: ", round(t2-t1,2))

nruns=5
for fn in ['FCItdm12kern_a', 'FCItdm12kern_b', 'FCItdm12kern_ab', 'FCIrdm12kern_sf']: 
  performance_checker(fn, cibra, ciket, norb, nelec, link_index, nruns=nruns)
