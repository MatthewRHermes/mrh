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

norb = 12
nelec = 9

def run_test(cre, norb, nelec, spin, reorder):
    nelec_copy = list(_unpack_nelec(nelec))
    print(cre, nelec, spin, reorder)
    if not cre:
        nelec_copy[spin] -=1
    nelec_ket = _unpack_nelec(nelec_copy)
    nelec_bra = [x for x in nelec_copy]
    nelec_bra[spin] += 1
    
    na_bra = math.comb(norb, nelec_bra[0])
    nb_bra = math.comb(norb, nelec_bra[1])
    na_ket = math.comb(norb, nelec_ket[0])
    nb_ket = math.comb(norb, nelec_ket[1])
    #cibra = np.arange(na_bra*nb_bra).reshape(na_bra, nb_bra)+0.5
    #ciket = np.arange(na_ket*nb_ket).reshape(na_ket, nb_ket)+4.5
    cibra = np.random.random((na_bra, nb_bra))
    ciket = np.random.random((na_ket, nb_ket))
    #print(ciket.shape, cibra.shape)
    #print(ciket)
    #print(cibra) 
    if not cre: 
        cibra, ciket = ciket, cibra

    _trans_rdm13hs(cre, cibra , ciket, norb, nelec, spin, None, reorder)


[run_test(cre, norb, nelec, spin, reorder) for cre in range(2) for spin in range(2) for reorder in range(2)]

#nelec = _unpack_nelec(nelec)
#neleca, nelecb = _unpack_nelec(nelec)
#na = math.comb(norb, neleca-1)
#nb = math.comb(norb, nelecb)
#cibra = np.arange(na*nb).reshape(na,nb)+0.0
#cibra = np.random.random((na,nb))
#na = math.comb(norb, neleca)
#ciket = np.arange(na*nb).reshape(na,nb)+100.0
#ciket = np.random.random((na,nb))
#link_indexa = cistring.gen_linkstr_index(range(norb+1), neleca)
#link_indexb = cistring.gen_linkstr_index(range(norb+1), nelecb)
#link_index = (link_indexa, link_indexb)
#link_index=None
#_trans_rdm13hs(False, cibra, ciket, norb, nelec, spin=0, link_index = link_index, reorder=False)
#_trans_rdm13hs(True, cibra, ciket, norb, nelec, spin=0, link_index = link_index, reorder=True)

#norb = 10
#nelec = 9
#neleca, nelecb = _unpack_nelec(nelec)
#na = math.comb(norb, neleca)
#nb = math.comb(norb, nelecb-1)
#cibra = np.random.random((na,nb))
#nb = math.comb(norb, nelecb)
#ciket = np.random.random((na,nb))
#link_indexa = cistring.gen_linkstr_index(range(norb+1), neleca)
#link_indexb = cistring.gen_linkstr_index(range(norb+1), nelecb)
#link_index = (link_indexa, link_indexb)
#link_index=None

#_trans_rdm13hs(False, cibra, ciket, norb, nelec, spin=1, link_index = link_index, reorder=False)
