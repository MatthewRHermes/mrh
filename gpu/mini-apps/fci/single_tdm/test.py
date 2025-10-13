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
from mrh.my_pyscf.fci.rdm import _unpack
from mrh.my_pyscf.fci import dummy
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
  rdm.make_rdm12_spin1('FCItdm12kern_ab', cibra, ciket, norb, nelec)

norb, nelec = 4, 4
run_test(norb, nelec)

norb, nelec = 10, 7
run_test(norb, nelec)


norb = 3
nelec = 3


def run_test_v2(norb, nelec):
    nelec_ket = _unpack_nelec(nelec)
    nelec_bra = list(_unpack_nelec(nelec))
    nelec_bra[0] +=1
    nelec_bra[1] -=1
    na_bra = math.comb(norb, nelec_bra[0])
    nb_bra = math.comb(norb, nelec_bra[1])
    na_ket = math.comb(norb, nelec_ket[0])
    nb_ket = math.comb(norb, nelec_ket[1])
    cibra = np.arange(na_bra*nb_bra).reshape(na_bra, nb_bra)+1.0
    ciket = np.arange(na_ket*nb_ket).reshape(na_ket, nb_ket)-1.0
    #cibra = np.random.random((na_bra, nb_bra))
    #ciket = np.random.random((na_ket, nb_ket))
    nelec_ket = _unpack_nelec (nelec)
    nelec_bra = list (_unpack_nelec (nelec))
    nelec_bra[0] += 1
    nelec_bra[1] -= 1
    nelecd = [nelec_bra[0], nelec_ket[1]]
    linkstr = _unpack (norb+1, nelecd, None)
    errmsg = ("For the spin-flip transition density matrix functions, the linkstr must be for "
              "(neleca+1,nelecb) electrons occupying norb+1 orbitals.")
    for i in range (2): assert (linkstr[i].shape[1]==(nelecd[i]*(norb-nelecd[i]+2))), errmsg
    ciket = dummy.add_orbital (ciket, norb, nelec_ket, occ_a=1, occ_b=0)
    cibra = dummy.add_orbital (cibra, norb, nelec_bra, occ_a=0, occ_b=1)
    fn = 'FCItdm12kern_ab'
    dm2dum = rdm.make_rdm12_spin1 (fn, ciket, cibra, norb+1, nelecd, linkstr, 0)[1]
     
run_test_v2(norb, nelec)
