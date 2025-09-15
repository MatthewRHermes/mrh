gpu_run=1
if gpu_run:from mrh.my_pyscf.gpu import libgpu
import pyscf
import numpy as np 
if gpu_run:from gpu4mrh import patch_pyscf
from pyscf import gto, scf, tools, mcscf, lib
from pyscf.fci import rdm, cistring
from pyscf.fci.addons import _unpack_nelec
from mrh.my_pyscf.mcscf.lasscf_async import LASSCF
import math
if gpu_run:
  gpu = libgpu.init()
  from pyscf.lib import param
  param.use_gpu = gpu
  param.gpu_debug=True
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

def run_test(norb, nelec):
  neleca, nelecb = _unpack_nelec(nelec)
  na = math.comb(norb, neleca)
  nb = math.comb(norb, nelecb)
  cibra = np.random.random((na,nb))
  ciket = np.random.random((na,nb))
  #rdm.make_rdm12_spin1('FCItdm12kern_a', cibra, ciket, norb, nelec)
  #rdm.make_rdm12_spin1('FCItdm12kern_b', cibra, ciket, norb, nelec)
  rdm.make_rdm12_spin1('FCItdm12kern_ab', cibra, ciket, norb, nelec)
  #rdm.make_rdm12_spin1('FCIrdm12kern_sf', cibra, ciket, norb, nelec)

norb, nelec = 4, 4
run_test(norb, nelec)

norb, nelec = 10, 7
run_test(norb, nelec)
