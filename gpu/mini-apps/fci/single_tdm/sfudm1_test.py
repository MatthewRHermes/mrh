gpu_run=1
if gpu_run:from mrh.my_pyscf.gpu import libgpu
import pyscf
import numpy as np 
if gpu_run:from gpu4mrh import patch_pyscf
from pyscf import gto, scf, lib
from pyscf.fci import  cistring
from pyscf.fci.addons import _unpack_nelec
import math
from mrh.my_pyscf.fci.rdm import trans_sfudm1 

if gpu_run:
  gpu = libgpu.init()
  from pyscf.lib import param
  param.use_gpu = gpu
  #param.gpu_debug=True
  param.custom_fci=True
  param.custom_debug = True
lib.logger.TIMER_LEVEL=lib.logger.INFO

geom = ''' Mg 0 0 0;
           O 0 0 2;
           Mg 0 0 4;
           O 0 0 6;
           Mg 0 0 8;
           O  0 0 10;
           Mg 0 0 12;
           O  0 0 14;
           Mg 0 0 16;
           O  0 0 18;
           Mg 0 0 24;
           O  0 0 26;'''

mol = gto.M(atom=geom, basis='def2-SVP', verbose=4)

mol.output='test.log'
mol.build()

mf = scf.RHF(mol)
mf=mf.density_fit()
#mf.with_df.auxbasis = pyscf.df.make_auxbasis(mol)
mf.max_cycle=1
mf.kernel()

def run_test(norb, nelec):
    nelec_ket = _unpack_nelec(nelec)
    nelec_bra = list(_unpack_nelec(nelec))
    nelec_bra[0] +=1
    nelec_bra[1] -=1
    print(nelec_bra, nelec_ket)
    na_bra = math.comb(norb, nelec_bra[0])
    nb_bra = math.comb(norb, nelec_bra[1])
    na_ket = math.comb(norb, nelec_ket[0])
    nb_ket = math.comb(norb, nelec_ket[1])
    #cibra = np.arange(na_bra*nb_bra).reshape(na_bra, nb_bra)+1.0
    #ciket = np.arange(na_ket*nb_ket).reshape(na_ket, nb_ket)-1.0
    cibra = np.random.random((na_bra, nb_bra))
    ciket = np.random.random((na_ket, nb_ket))
    print(na_bra, nb_bra, na_ket, nb_ket)
    trans_sfudm1 (cibra, ciket, norb, nelec) 

norb, nelec = 18, (17,10)
run_test(norb, nelec)
#norb, nelec = 11, 15
#run_test(norb, nelec)
