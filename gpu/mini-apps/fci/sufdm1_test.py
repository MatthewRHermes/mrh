gpu_run=1
if gpu_run:from mrh.my_pyscf.gpu import libgpu
import pyscf
import numpy as np 
if gpu_run:from gpu4mrh import patch_pyscf
from pyscf import gto, scf, tools, mcscf, lib
from pyscf.fci import  cistring
from mrh.my_pyscf.fci import _unpack
from pyscf.fci.addons import _unpack_nelec
from mrh.my_pyscf.mcscf.lasscf_async import LASSCF
import math

if gpu_run:
  gpu = libgpu.init()
  from pyscf.lib import param
  param.use_gpu = gpu
  #param.gpu_debug=True
  param.custom_fci=True
  param.custom_debug = True
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
nelec_ket = _unpack_nelec(nelec)
nelec_bra = _unpack_nelec(nelec)
nelec_bra[0] +=1
nelec_bra[0] -=1
nelecd = [nelec_bra[0], nelec_ket[1]]
linkstr = _unpack(norb+1, nelecd)
na, nlinka = linkstr[0].shape[:2] 
nb, nlinkb = linkstr[1].shape[:2] 
cibra = np.random.random
