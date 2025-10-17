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

def multi_gpu_loop(cre, bravecs, ketvecs,norb,nelec, spin, reorder):
  #print("Starting multiGPU")
  from mrh.my_pyscf.gpu import libgpu
  from pyscf.lib import param
  gpu=param.use_gpu
  nelec = list (_unpack_nelec (nelec))
  cre = False #doing this because that's what the main function is
  if not cre:
      #bravecs, ketvecs = ketvecs, bravecs
      nelec[spin] -= 1

  nelec_ket = _unpack_nelec (nelec)
  nelec_bra = [x for x in nelec]
  nelec_bra[spin] += 1
  linkstr = _unpack (norb+1, nelec_bra, None)
  errmsg = ("For the half-particle transition density matrix functions, the linkstr must "
            "be for nelec+1 electrons occupying norb+1 orbitals.")
  for i in range (2): assert (linkstr[i].shape[1]==(nelec_bra[i]*(norb-nelec_bra[i]+2))), errmsg
  ia_ket, ja_ket, ib_ket, jb_ket, sgn_ket = dummy.dummy_orbital_params(norb, nelec_ket, occ_a = (1-spin), occ_b = spin)
  ia_bra, ja_bra, ib_bra, jb_bra, sgn_bra = dummy.dummy_orbital_params(norb, nelec_bra, occ_a = 0, occ_b = 0)
  na, nlinka = linkstr[0].shape[:2] 
  nb, nlinkb = linkstr[1].shape[:2] 
  n_bra, na_bra, nb_bra = bravecs.shape
  n_ket, na_ket, nb_ket = ketvecs.shape

  libgpu.init_tdm1(gpu, norb+1)
  libgpu.init_tdm3hab(gpu, norb+1)
  size_tdm1h = norb
  size_tdm1h_full = n_bra*n_ket*size_tdm1h
  size_tdm3h = 2*norb*norb*norb
  size_tdm3h_full = n_bra*n_ket*size_tdm3h
  libgpu.init_tdm1_host(gpu, size_tdm1h_full)
  libgpu.init_tdm2_host(gpu, size_tdm3h_full)
  libgpu.push_link_index_ab(gpu, na, nb, nlinka, nlinkb, linkstr[0], linkstr[1])
  libgpu.copy_bravecs_host(gpu, bravecs, n_bra, na_bra, nb_bra) 
  libgpu.copy_ketvecs_host(gpu, ketvecs, n_ket, na_ket, nb_ket) 
  for count, (j, i) in enumerate(product (range(n_ket), range(n_bra))):
    libgpu.push_cibra_from_host(gpu, i, na_bra, nb_bra, count)
    libgpu.push_ciket_from_host(gpu, j, na_ket, nb_ket, count)
    libgpu.compute_tdm13h_spin_v4(gpu, na, nb, nlinka, nlinkb, norb+1, spin, reorder,
                                 ia_bra, ja_bra, ib_bra, jb_bra, sgn_bra,
                                 ia_ket, ja_ket, ib_ket, jb_ket, sgn_ket, count) #TODO: write a better name
    if reorder: libgpu.reorder_rdm(gpu, norb+1, count)
    libgpu.pull_tdm3hab_v2_host(gpu, i, j, n_bra, n_ket, norb, cre, spin, count)

  #tdm1h = np.zeros ((ketvecs.shape[0],bravecs.shape[0],norb), dtype=bravecs.dtype)
  #tdm3h = np.zeros ((ketvecs.shape[0],bravecs.shape[0],2,norb,norb,norb), dtype=bravecs.dtype)
  tdm1h = np.zeros ((bravecs.shape[0],ketvecs.shape[0],norb), dtype=bravecs.dtype)
  tdm3h = np.zeros ((bravecs.shape[0],ketvecs.shape[0],2,norb,norb,norb), dtype=bravecs.dtype)
  libgpu.copy_tdm1_host_to_page(gpu, tdm1h, size_tdm1h_full) 
  libgpu.copy_tdm2_host_to_page(gpu, tdm3h, size_tdm3h_full) 
  return tdm1h, tdm3h
  
def o0_loop(cre, bravecs, ketvecs,norb,nelec, spin, reorder):
  from mrh.my_pyscf.fci.rdm import trans_rdm13ha_des, trans_rdm13hb_des #is make_rdm12_spin1
  trans_rdm13h = (trans_rdm13ha_des, trans_rdm13hb_des)[spin]
  tdm1h = np.zeros ((bravecs.shape[0],ketvecs.shape[0],norb), dtype=bravecs.dtype)
  tdm3h = np.zeros ((bravecs.shape[0],ketvecs.shape[0],2,norb,norb,norb), dtype=bravecs.dtype)
  for i, j in product (range (bravecs.shape[0]), range (ketvecs.shape[0])):
    #d1s, d2s = trans_rdm13h (bravecs[i], ketvecs[j], norb, nelec, link_index=None)
    d1s, d2s = trans_rdm13h (ketvecs[j], bravecs[i], norb, nelec, link_index=None)
    tdm1h[i,j] = d1s
    tdm3h[i,j] = np.stack (d2s, axis=0)
  return tdm1h, tdm3h



def test_tdm13h_loop(cre, n_bra, n_ket, norb, nelec, spin, reorder):
    print('start cre:',cre, 'nelec:', nelec, 'spin:',spin, 'reorder:',reorder)
    nelec_copy = list(_unpack_nelec(nelec))
    cre = False
    if not cre:
        nelec_copy[spin] -=1
    nelec_ket = _unpack_nelec(nelec_copy)
    nelec_bra = [x for x in nelec_copy]
    nelec_bra[spin] += 1
    
    na_bra = math.comb(norb, nelec_bra[0])
    nb_bra = math.comb(norb, nelec_bra[1])
    na_ket = math.comb(norb, nelec_ket[0])
    nb_ket = math.comb(norb, nelec_ket[1])

    bravecs = np.empty((n_bra, na_bra, nb_bra))
    ketvecs = np.empty((n_ket, na_ket, nb_ket))
    #for _nbra in range(n_bra): bravecs[_nbra] = np.random.random((na_bra, nb_bra))
    #for _nket in range(n_ket): ketvecs[_nket] = np.random.random((na_ket, nb_ket))
    for _nbra in range(n_bra): bravecs[_nbra] = np.arange(na_bra* nb_bra).reshape((na_bra, nb_bra)) + 0.5 + _nbra
    for _nket in range(n_ket): ketvecs[_nket] = np.arange(na_ket* nb_ket).reshape((na_ket, nb_ket)) - 0.5 - _nket
    
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
      tdm1h, tdm3h = o0_loop(cre, bravecs, ketvecs,norb,nelec, spin, reorder)
      tdm1h_c, tdm3h_c = multi_gpu_loop(cre, bravecs, ketvecs,norb, nelec, spin, reorder)
      tdm1h_correct = np.allclose(tdm1h, tdm1h_c)
      tdm3h_correct = np.allclose(tdm3h, tdm3h_c)
      if tdm1h_correct and tdm3h_correct: 
        print('TDM13h loop calculated correctly')
      else:
        print('TDM13h loop incorrect')
        print('TDM1h correct?', tdm1h_correct)
        print('TDM3h correct?', tdm3h_correct)
        diff = tdm3h-tdm3h_c
        #print(tdm3h)
        #print(tdm3h_c)
        for i, j in product (range (bravecs.shape[0]), range (ketvecs.shape[0])):
          if np.sum([diff[i,j]!=0]): 
            print('bra:',i, 'ket',j, False)
            #print(tdm3h[i,j])
            #print(tdm3h_c[i,j])
          else: print('bra:',i, 'ket',j, True)
          #print(i,j,if False in [diff[i,j]==0]: )
        exit()
    elif custom_fci and use_gpu and mgpu_fci: 
      print("in gpu channel")
      tdm1h, tdm3h = multi_gpu_loop(cre, bravecs, ketvecs,norb,nelec, spin, reorder)
    else:
      print("in cpu channel")
      tdm1h, tdm3h = o0_loop(cre, bravecs, ketvecs,norb,nelec, spin, reorder)
    return tdm1h, tdm3h

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

  norb, nelec,  n_bra, n_ket = 11,14, 5,4
  [test_tdm13h_loop(cre, n_bra, n_ket,norb, nelec, spin, reorder) for cre in range(2) for spin in range(2) for reorder in range(2)]
  #[test_tdm13h_loop(cre, n_bra, n_ket,norb, nelec, spin, reorder) for cre in range(2) for spin in range( 2) for reorder in range(2)]
  [test_tdm13h_loop(cre, n_bra, n_ket,norb, nelec, spin, reorder) for cre in range(2) for spin in range(1,-1,-1) for reorder in range(2)]
  #norb, nelec,  n_bra, n_ket = 8,14, 4,5
  #[test_tdm13h_loop(cre, n_bra, n_ket,norb, nelec, spin, reorder) for cre in range(2) for spin in range( 2) for reorder in range(2)]
  #norb, nelec,  n_bra, n_ket = 8,6, 4,5
  #[test_tdm13h_loop(cre, n_bra, n_ket,norb, nelec, spin, reorder) for cre in range(2) for spin in range( 2) for reorder in range(2)]
  #norb, nelec,  n_bra, n_ket = 4,4, 2,3
  [test_tdm13h_loop(cre, n_bra, n_ket,norb, nelec, spin, reorder) for cre in range(2) for spin in range(2) for reorder in range(2)]
  #norb, nelec = 10, 13
  #n_bra, n_ket = 6,5
  #[test_tdm13h_loop(cre, n_bra, n_ket,norb, nelec, spin, reorder) for cre in range(2) for spin in range( 2) for reorder in range(2)]
  if gpu_run: libgpu.destroy_device(gpu)

