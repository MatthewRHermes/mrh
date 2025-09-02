gpu_run=1
if gpu_run:from mrh.my_pyscf.gpu import libgpu
import pyscf
import numpy as np 
if gpu_run:from gpu4mrh import patch_pyscf
from pyscf import gto, scf, lib
from pyscf.fci import  cistring
from pyscf.fci.addons import _unpack_nelec
import math
from mrh.my_pyscf.fci.rdm import trans_ppdm

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
nelec = 8
nelec_ket = _unpack_nelec(nelec)
nelec_bra = _unpack_nelec(nelec)


def run_test(norb, nelec, spin):
    s1 = int (spin>1)
    s2 = int (spin>0)
    ndum = 2 - (spin%2)
    nelec_ket = _unpack_nelec (nelec)
    nelec_bra = list (_unpack_nelec (nelec))
    nelec_bra[s1] += 1
    nelec_bra[s2] += 1
    occ_a, occ_b = int (spin<2), int (spin>0)
    #linkstr = _unpack (norb+ndum, nelec_bra, link_index)
    #nelecd = [nelec_ket[0], nelec_ket[1]]
    na_bra = math.comb(norb, nelec_bra[0])
    nb_bra = math.comb(norb, nelec_bra[1])
    na_ket = math.comb(norb, nelec_ket[0])
    nb_ket = math.comb(norb, nelec_ket[1])
    cibra = np.arange(na_bra*nb_bra).reshape(na_bra, nb_bra)+0.5
    ciket = np.arange(na_ket*nb_ket).reshape(na_ket, nb_ket)+4.5
    #cibra = np.random.random((na_bra, nb_bra))
    #ciket = np.random.random((na_ket, nb_ket))
    trans_ppdm (cibra, ciket, norb, nelec, spin = spin) 

#nelecd = [nelec_bra[0], nelec_ket[1]]
#linkstr = _unpack(norb+1, nelecd)
#na, nlinka = linkstr[0].shape[:2] 
#nb, nlinkb = linkstr[1].shape[:2] 

[run_test(norb, nelec, i) for i in range(3)]
