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
from mrh.my_pyscf.fci.rdm import trans_hhdm
from itertools import product

def multi_gpu_loop(bravecs, ketvecs, norb, nelec, spin):
  from mrh.my_pyscf.gpu import libgpu
  from pyscf.lib import param
  gpu=param.use_gpu

  nelec_copy = list(_unpack_nelec (nelec))
  nelec_copy[int (spin>1)] -= 1
  nelec_copy[int (spin>0)] -= 1

  s1 = int (spin>1)
  s2 = int (spin>0)
  ndum = 2 - (spin%2)
  nelec_ket = _unpack_nelec (nelec_copy)
  nelec_bra = list (_unpack_nelec (nelec_copy))
  nelec_bra[s1] += 1
  nelec_bra[s2] += 1
  occ_a, occ_b = int (spin<2), int (spin>0)
  linkstr = _unpack (norb+ndum, nelec_bra, None)
  errmsg = ("For the pair-creation transition density matrix functions, the linkstr must "
          "be for nelec+2 electrons occupying norb+1/norb+2 (ab/other spin case) orbitals.")
  assert (linkstr[0].shape[1]==(nelec_bra[0]*(norb+ndum-nelec_bra[0]+1))), errmsg
  assert (linkstr[1].shape[1]==(nelec_bra[1]*(norb+ndum-nelec_bra[1]+1))), errmsg
  nelecd = [nelec_ket[0], nelec_ket[1]]
  nelecd_copy = nelecd.copy()
  n_bra, na_bra, nb_bra = bravecs.shape
  n_ket, na_ket, nb_ket = ketvecs.shape
  ia_bra = ia_ket = ib_bra = ib_ket = 0
  ja_bra, jb_bra, ja_ket, jb_ket = na_bra, nb_bra, na_ket, nb_ket
  sgn_bra = sgn_ket = 1
  for i in range (ndum):
    ia_ket_new, ja_ket_new, ib_ket_new, jb_ket_new, sgn_ket_new = dummy.dummy_orbital_params(norb+i, nelecd_copy, occ_a, occ_b)
    nelecd_copy[0] +=occ_a
    nelecd_copy[1] +=occ_b 
    ia_bra_new, ja_bra_new, ib_bra_new, jb_bra_new, sgn_bra_new = dummy.dummy_orbital_params(norb+i, nelec_bra, 0, 0)
    ia_bra += ia_bra_new
    ib_bra += ib_bra_new
    ia_ket += ia_ket_new
    ib_ket += ib_ket_new
    ja_bra = ia_bra + na_bra
    jb_bra = ib_bra + nb_bra
    ja_ket = ia_ket + na_ket
    jb_ket = ib_ket + nb_ket
    sgn_bra *= sgn_bra_new
    sgn_ket *= sgn_ket_new
  na, nlinka = linkstr[0].shape[:2] 
  nb, nlinkb = linkstr[1].shape[:2] 
  libgpu.init_tdm1(gpu, norb)
  libgpu.init_tdm2(gpu, norb+ndum)
  libgpu.push_link_index_ab(gpu, na, nb, nlinka, nlinkb, linkstr[0], linkstr[1])
  size_hhdm = norb*norb
  size_hhdm_full = n_bra*n_ket*size_hhdm
  libgpu.init_tdm1_host(gpu, size_hhdm_full)
  libgpu.copy_bravecs_host(gpu, bravecs, n_bra, na_bra, nb_bra) 
  libgpu.copy_ketvecs_host(gpu, ketvecs, n_ket, na_ket, nb_ket) 
  for count, (i, j) in enumerate(product (range(n_bra), range(n_ket))):
    libgpu.push_cibra_from_host(gpu, i, na_bra, nb_bra, count)
    libgpu.push_ciket_from_host(gpu, j, na_ket, nb_ket, count)
    libgpu.compute_tdmpp_spin_v4(gpu, na, nb, nlinka, nlinkb, norb+ndum, spin, 
                                 ia_bra, ja_bra, ib_bra, jb_bra, sgn_bra, 
                                 ia_ket, ja_ket, ib_ket, jb_ket, sgn_ket, count) 
    libgpu.pull_tdm1_host(gpu, j, i, n_ket, n_bra, size_hhdm, 1, count)#remember that hhdm is norb*norb, so dm1 is fine.
  hhdm = np.zeros ((ketvecs.shape[0],bravecs.shape[0],norb,norb), dtype=bravecs.dtype)
  libgpu.copy_tdm1_host_to_page(gpu, hhdm, size_hhdm_full) 
  return hhdm.transpose(0,1,3,2)
  
def o0_loop(bravecs, ketvecs, norb, nelec_ket, spin):
  hhdm = np.zeros ((bravecs.shape[0],ketvecs.shape[0],norb,norb), dtype=bravecs.dtype)
  for i, j in product (range (bravecs.shape[0]), range (ketvecs.shape[0])):
    d1 = trans_hhdm (bravecs[i], ketvecs[j], norb, nelec_ket, spin=spin, link_index=None)
    hhdm[i,j] = d1
  return hhdm

def test_hhdm_loop(n_bra, n_ket, norb, nelec, spin):

    nelec_copy = list(_unpack_nelec (nelec))
    nelec_copy[int (spin>1)] -= 1
    nelec_copy[int (spin>0)] -= 1

    s1 = int (spin>1)
    s2 = int (spin>0)
    ndum = 2 -(spin%2)
    nelec_ket = _unpack_nelec (nelec_copy)
    nelec_bra = list (_unpack_nelec (nelec_copy))
    nelec_bra[s1] += 1
    nelec_bra[s2] += 1
    occ_a, occ_b = int (spin<2), int (spin>0)
    print(nelec_bra, nelec_ket)
    na_bra = math.comb(norb, nelec_bra[0])
    nb_bra = math.comb(norb, nelec_bra[1])
    na_ket = math.comb(norb, nelec_ket[0])
    nb_ket = math.comb(norb, nelec_ket[1])
    bravecs = np.empty((n_bra, na_bra, nb_bra))
    ketvecs = np.empty((n_ket, na_ket, nb_ket))
    for _nbra in range(n_bra): bravecs[_nbra] = np.random.random((na_bra, nb_bra))
    for _nket in range(n_ket): ketvecs[_nket] = np.random.random((na_ket, nb_ket))
    #for _nbra in range(n_bra): bravecs[_nbra] = np.arange(na_bra* nb_bra).reshape((na_bra, nb_bra)) + 0.5
    #for _nket in range(n_ket): ketvecs[_nket] = np.arange(na_ket* nb_ket).reshape((na_ket, nb_ket)) + 0.5
    
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
      tdmhh_c = multi_gpu_loop(bravecs, ketvecs,norb,nelec,spin)
      tdmhh = o0_loop(ketvecs, bravecs, norb,nelec,spin)
      tdmhh_correct = np.allclose(tdmhh, tdmhh_c)
      if tdmhh_correct: 
        print('TDMhh loop calculated correctly')
      else:
        print('TDMhh loop incorrect')
        print(tdmhh.shape)
        print(tdmhh_c.shape)
        diff = tdmhh-tdmhh_c
        print(diff.shape)
        print(diff)
 
        #print(tdmhh)
        #print(tdmhh_c)
        exit()
    elif custom_fci and use_gpu and mgpu_fci: 
      print("in gpu channel")
      tdmhh = multi_gpu_loop(bravecs, ketvecs,norb,nelec,spin)
    else:
      print("in cpu channel")
      tdmhh = o0_loop(bravecs, ketvecs,norb,nelec,spin)
    return tdmhh

if __name__=="__main__":
  if gpu_run:
    gpu = libgpu.init()
    from pyscf.lib import param
    param.use_gpu = gpu
    param.custom_fci=True
    param.mgpu_fci=True
    param.mgpu_fci_debug=True
    param.custom_debug = False
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

  norb, nelec = 11, 15
  n_bra, n_ket = 17,1
  #norb, nelec = 8, 9
  #n_bra, n_ket = 3,3
  [test_hhdm_loop(n_bra, n_ket,norb, nelec, i) for i in range(3)]
  if gpu_run: libgpu.destroy_device(gpu)

