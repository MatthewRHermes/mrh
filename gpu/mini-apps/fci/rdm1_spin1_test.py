gpu_run=1
if gpu_run:from mrh.my_pyscf.gpu import libgpu
import pyscf
import numpy as np 
if gpu_run:from gpu4mrh import patch_pyscf
from pyscf import gto, scf, tools, mcscf, lib
from pyscf.fci import rdm, cistring
from pyscf.fci.addons import _unpack_nelec
from mrh.my_pyscf.mcscf.lasscf_async import LASSCF
if gpu_run:
  gpu = libgpu.init()
  from pyscf.lib import param
  param.use_gpu = gpu
  param.gpu_debug = True
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

norb = 10
nelec = 9

neleca, nelecb = _unpack_nelec(nelec)
link_indexa = link_indexb = cistring.gen_linkstr_index(range(norb), neleca)
if neleca != nelecb: link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)

na = link_indexa.shape[0]
nb = link_indexb.shape[0]
cibra = np.random.random((na,nb))
ciket = np.random.random((na,nb))
link_index = (link_indexa, link_indexb)
rdm.make_rdm1_spin1('FCItrans_rdm1a', cibra, ciket, norb, nelec, link_index)#, use_gpu = True, gpu=gpu)
rdm.make_rdm1_spin1('FCItrans_rdm1b', cibra, ciket, norb, nelec, link_index)#, use_gpu = True, gpu=gpu)
rdm.make_rdm1_spin1('FCImake_rdm1a', cibra, ciket, norb, nelec, link_index)#, use_gpu = True, gpu=gpu)
rdm.make_rdm1_spin1('FCImake_rdm1b', cibra, ciket, norb, nelec, link_index)#, use_gpu = True, gpu=gpu)
