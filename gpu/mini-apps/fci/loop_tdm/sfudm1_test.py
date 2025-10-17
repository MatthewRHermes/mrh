gpu_run=1
if gpu_run:from mrh.my_pyscf.gpu import libgpu
import pyscf
import numpy as np 
if gpu_run:from gpu4mrh import patch_pyscf
from pyscf import gto, scf, lib
from pyscf.fci import  cistring
from pyscf.fci.addons import _unpack_nelec
from mrh.my_pyscf.fci.rdm import _unpack
from mrh.my_pyscf.fci import dummy
import math
from mrh.my_pyscf.fci.rdm import trans_sfudm1 
from itertools import product

def multi_gpu_loop(bravecs, ketvecs, norb, nelec):
  from mrh.my_pyscf.gpu import libgpu
  from pyscf.lib import param
  gpu=param.use_gpu
  nelec_ket = _unpack_nelec (nelec)
  nelec_bra = list (_unpack_nelec (nelec))
  nelec_bra[0] += 1
  nelec_bra[1] -= 1
  nelecd = [nelec_bra[0], nelec_ket[1]]
  linkstr = _unpack (norb+1, nelecd, None)
  errmsg = ("For the spin-flip transition density matrix functions, the linkstr must be for "
            "(neleca+1,nelecb) electrons occupying norb+1 orbitals.")
  for i in range (2): assert (linkstr[i].shape[1]==(nelecd[i]*(norb-nelecd[i]+2))), errmsg
  ia_ket, ja_ket, ib_ket, jb_ket, sgn_ket = dummy.dummy_orbital_params(norb, nelec_ket, occ_a = 1, occ_b = 0)
  ia_bra, ja_bra, ib_bra, jb_bra, sgn_bra = dummy.dummy_orbital_params(norb, nelec_bra, occ_a = 0, occ_b = 1)
  na, nlinka = linkstr[0].shape[:2] 
  nb, nlinkb = linkstr[1].shape[:2] 
  n_bra, na_bra, nb_bra = bravecs.shape
  n_ket, na_ket, nb_ket = ketvecs.shape
  libgpu.init_tdm1(gpu, norb)
  libgpu.init_tdm2(gpu, norb+1)
  libgpu.push_link_index_ab(gpu, na, nb, nlinka, nlinkb, linkstr[0], linkstr[1])
  size_sfudm = norb*norb
  size_sfudm_full = n_bra*n_ket*size_sfudm
  libgpu.push_link_index_ab(gpu, na, nb, nlinka, nlinkb, linkstr[0], linkstr[1])
  libgpu.init_tdm1_host(gpu, size_sfudm_full)
  ##From here on out, remember ket and bra are switched
  libgpu.copy_ketvecs_host(gpu, bravecs, n_bra, na_bra, nb_bra) 
  libgpu.copy_bravecs_host(gpu, ketvecs, n_ket, na_ket, nb_ket) 
  for count, (i, j) in enumerate(product (range(n_bra), range(n_ket))):
    libgpu.push_cibra_from_host(gpu, j, na_ket, nb_ket, count)
    libgpu.push_ciket_from_host(gpu, i, na_bra, nb_bra, count)
    libgpu.compute_sfudm_v2(gpu, na, nb, nlinka, nlinkb, norb+1, 
                       ia_ket, ja_ket, ib_ket, jb_ket, sgn_ket,
                       ia_bra, ja_bra, ib_bra, jb_bra, sgn_bra, count)
    libgpu.pull_tdm1_host(gpu, j, i, n_ket, n_bra, size_sfudm, 1, count)#remember that hhdm is norb*norb, so dm1 is fine.
  sfudm = np.zeros ((bravecs.shape[0],ketvecs.shape[0],norb,norb), dtype=bravecs.dtype)
  libgpu.copy_tdm1_host_to_page(gpu, sfudm, size_sfudm_full) 
  return sfudm
  
def o0_loop(bravecs, ketvecs, norb, nelec_ket):
  sfudm = np.zeros ((bravecs.shape[0],ketvecs.shape[0],norb,norb), dtype=bravecs.dtype)
  for i, j in product (range (bravecs.shape[0]), range (ketvecs.shape[0])):
    d1 = trans_sfudm1 (bravecs[i], ketvecs[j], norb, nelec_ket, link_index=None)
    sfudm[i,j] = d1
  return sfudm



def test_sfudm_loop(n_bra, n_ket, norb, nelec):
    nelec_ket = _unpack_nelec(nelec)
    nelec_bra = list(_unpack_nelec(nelec))
    nelec_bra[0] +=1
    nelec_bra[1] -=1
    na_bra = math.comb(norb, nelec_bra[0])
    nb_bra = math.comb(norb, nelec_bra[1])
    na_ket = math.comb(norb, nelec_ket[0])
    nb_ket = math.comb(norb, nelec_ket[1])

    bravecs = np.empty((n_bra, na_bra, nb_bra))
    ketvecs = np.empty((n_ket, na_ket, nb_ket))
    #for _nbra in range(n_bra): bravecs[_nbra] = np.random.random((na_bra, nb_bra))
    #for _nket in range(n_ket): ketvecs[_nket] = np.random.random((na_ket, nb_ket))
    for _nbra in range(n_bra): bravecs[_nbra] = np.arange(na_bra* nb_bra).reshape((na_bra, nb_bra)) + 0.5
    for _nket in range(n_ket): ketvecs[_nket] = np.arange(na_ket* nb_ket).reshape((na_ket, nb_ket)) - 0.5
    
    from pyscf.lib import param
    try: 
      use_gpu = param.use_gpu
      gpu = use_gpu
    except: 
      use_gpu = None
    try: gpu_debug = param.gpu_debug
    except: gpu_debug = False
    try: custom_fci = param.custom_fci
    except: custom_fci = False
    try: custom_debug = param.custom_debug
    except: custom_debug = False
    try: mgpu_fci = param.mgpu_fci
    except: mgpu_fci = False
    try: mgpu_fci_debug = param.mgpu_fci_debug
    except: mgpu_fci_debug = False

    if mgpu_fci and mgpu_fci_debug and use_gpu:
      print("in debug channel")
      sfudm_c = multi_gpu_loop(bravecs, ketvecs,norb,nelec)
      sfudm = o0_loop(bravecs, ketvecs,norb,nelec)
      sfudm_correct = np.allclose(sfudm, sfudm_c)
      if sfudm_correct: 
        print('SFUDM1 loop calculated correctly')
      else:
        print('SFUDM1 loop incorrect')
        #print(tdmhh_c)
        #print(tdmhh)
        diff = sfudm-sfudm_c
        diff_index = np.nonzero(diff)
        print(diff.size)
        print(len(diff[diff!=0]))
        print(sfudm[diff!=0])
        print(sfudm_c[diff!=0])
        exit()
    elif custom_fci and use_gpu and mgpu_fci: 
      print("in gpu channel")
      sfudm = multi_gpu_loop(bravecs, ketvecs,norb,nelec)
    else:
      print("in cpu channel")
      sfudm = o0_loop(bravecs, ketvecs,norb,nelec)
    return sfudm

if __name__=="__main__":
  if gpu_run:
    gpu = libgpu.init()
    from pyscf.lib import param
    param.use_gpu = gpu
    param.custom_fci=True
    param.mgpu_fci=True
    param.mgpu_fci_debug=True
    #param.custom_debug = True
  lib.logger.TIMER_LEVEL=lib.logger.INFO

  geom = ''' K 0 0 0;
           K 0 0 2;
           K 0 0 4;
           K 0 0 8;
           K 0 0 10;
           K 0 0 12;'''
  basis = 'def2tzvp'
  if gpu_run: mol = gto.M(use_gpu = gpu, atom=geom, basis=basis, verbose=1)
  else: mol = gto.M(atom=geom, basis=basis, verbose=1)

  mol.output='test.log'
  mol.build()

  mf = scf.RHF(mol)
  mf=mf.density_fit()
  mf.with_df.auxbasis = pyscf.df.make_auxbasis(mol)
  mf.max_cycle=1
  mf.kernel()


  norb, nelec = 6, 5
  n_bra, n_ket = 10,4
  test_sfudm_loop(n_bra, n_ket,norb, nelec)
  norb, nelec = 11, 15
  n_bra, n_ket = 4,3
  test_sfudm_loop(n_bra, n_ket,norb, nelec)
  if gpu_run: libgpu.destroy_device(gpu)

