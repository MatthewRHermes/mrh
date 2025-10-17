gpu_run=1
if gpu_run:from mrh.my_pyscf.gpu import libgpu
import pyscf
import numpy as np 
if gpu_run:from gpu4mrh import patch_pyscf
from pyscf import gto, scf, lib
from pyscf.fci import  cistring
from pyscf.fci.addons import _unpack_nelec
from mrh.my_pyscf.fci.rdm import _unpack
import math
from pyscf.fci.direct_spin1 import trans_rdm12s
from itertools import product

def multi_gpu_loop(bravecs, ketvecs, norb, nelec, reorder=True):
  from mrh.my_pyscf.gpu import libgpu
  from pyscf.lib import param
  print('mgpu nelec',nelec)
  gpu=param.use_gpu
  linkstr = _unpack(norb, nelec,None) 
  na, nlinka = linkstr[0].shape[:2] 
  nb, nlinkb = linkstr[1].shape[:2] 
  n_bra, na_bra, nb_bra = bravecs.shape
  n_ket, na_ket, nb_ket = ketvecs.shape
  libgpu.init_tdm1(gpu, norb)
  libgpu.init_tdm2(gpu, norb)
  libgpu.push_link_index_ab(gpu, na, nb, nlinka, nlinkb, linkstr[0], linkstr[1])
  norb2 = norb*norb;
  size_tdm1 = norb2;
  size_tdm1_full = n_bra*n_ket*2*size_tdm1
  size_tdm2 = norb2*norb2;
  size_tdm2_full = n_bra*n_ket*4*size_tdm2;
  libgpu.init_tdm1_host(gpu, size_tdm1_full)
  libgpu.init_tdm2_host(gpu, size_tdm2_full)
  libgpu.copy_bravecs_host(gpu, bravecs, n_bra, na_bra, nb_bra) 
  libgpu.copy_ketvecs_host(gpu, ketvecs, n_ket, na_ket, nb_ket) 
  for count, (i, j) in enumerate(product (range(n_bra), range(n_ket))):
    #we pull DM1 = dm1a, dm1b, DM2 = dm2aa, dm2ab, dm2ba, dm2bb in the order

    libgpu.push_cibra_from_host(gpu, i, na_bra, nb_bra, count)
    libgpu.push_ciket_from_host(gpu, j, na_ket, nb_ket, count)

    #dm1a, dm2aa = rdm.make_rdm12_spin1('FCItdm12kern_a', cibra, ciket, norb, nelec, link_index, 2)
    libgpu.compute_tdm12kern_a_v2(gpu, na, nb, nlinka, nlinkb, norb, count)
    if reorder: libgpu.reorder_rdm(gpu, norb, count)
    libgpu.pull_tdm1_host(gpu, i, 2*j, n_bra, 2*n_ket, size_tdm1,2, count) #dm1a
    libgpu.pull_tdm2_host(gpu, i, 4*j, n_bra, 4*n_ket, size_tdm2,4, count) #dm2aa

    #dm1b, dm2bb = rdm.make_rdm12_spin1('FCItdm12kern_b', cibra, ciket, norb, nelec, link_index, 2)
    libgpu.compute_tdm12kern_b_v2(gpu, na, nb, nlinka, nlinkb, norb, count)
    if reorder: libgpu.reorder_rdm(gpu, norb, count)
    libgpu.pull_tdm1_host(gpu, i, 2*j+1, n_bra, 2*n_ket, size_tdm1,2, count) #dm1b
    libgpu.pull_tdm2_host(gpu, i, 4*j+3, n_bra, 4*n_ket, size_tdm2,4, count) #dm2bb

    #_, dm2ab = rdm.make_rdm12_spin1('FCItdm12kern_ab', cibra, ciket, norb, nelec, link_index, 0)
    libgpu.compute_tdm12kern_ab_v2(gpu, na, nb, nlinka, nlinkb, norb, count)
    libgpu.pull_tdm2_host(gpu, i, 4*j+1, n_bra, 4*n_ket, size_tdm2,4, count) #dm2ab

  libgpu.copy_ketvecs_host(gpu, bravecs, n_bra, na_bra, nb_bra) 
  libgpu.copy_bravecs_host(gpu, ketvecs, n_ket, na_ket, nb_ket) 
  for count, (i, j) in enumerate(product (range(n_bra), range(n_ket))):
    #split this is another loop
    #libgpu.push_cibra_from_host(gpu, i, na_bra, nb_bra, count)
    #libgpu.push_ciket_from_host(gpu, j, na_ket, nb_ket, count)
    libgpu.push_ciket_from_host(gpu, i, na_bra, nb_bra, count)
    libgpu.push_cibra_from_host(gpu, j, na_ket, nb_ket, count)
    libgpu.compute_tdm12kern_ab_v2(gpu, na, nb, nlinka, nlinkb, norb, count)
    #dm2ba = dm2ba.transpose(3,2,1,0)
    libgpu.transpose_tdm2(gpu, norb, count)
    libgpu.pull_tdm2_host(gpu, i, 4*j+2, n_bra, 4*n_ket, size_tdm2, 4,count) #dm2ba
  #remember, the tdm1 is transposed when return from direct_spin1, and then transposed again in trans_rdm12s_loop, so they can just be pulled.

  tdm1s = np.zeros ((bravecs.shape[0],ketvecs.shape[0],2,norb,norb), dtype=bravecs.dtype)
  tdm2s = np.zeros ((bravecs.shape[0],ketvecs.shape[0],4,norb,norb,norb,norb), dtype=bravecs.dtype)
  libgpu.copy_tdm1_host_to_page(gpu, tdm1s, size_tdm1_full)  
  libgpu.copy_tdm2_host_to_page(gpu, tdm2s, size_tdm2_full)  
  
  return tdm1s, tdm2s
  
def o0_loop(bravecs, ketvecs, norb, nelec):
  print('nelec',nelec)
  tdm1s = np.zeros ((bravecs.shape[0],ketvecs.shape[0],2,norb,norb), dtype=bravecs.dtype)
  tdm2s = np.zeros ((bravecs.shape[0],ketvecs.shape[0],4,norb,norb,norb,norb), dtype=bravecs.dtype)
  for i, j in product (range (bravecs.shape[0]), range (ketvecs.shape[0])):
    d1s, d2s = trans_rdm12s (bravecs[i], ketvecs[j], norb, nelec)
    # Transpose based on docstring of direct_spin1.trans_rdm12s
    tdm1s[i,j] = np.stack (d1s, axis=0).transpose (0, 2, 1)
    tdm2s[i,j] = np.stack (d2s, axis=0)
  return tdm1s, tdm2s



def test_tdm12_loop(n_bra, n_ket, norb, nelec):
    nelec_bra = list(_unpack_nelec(nelec))
    nelec_ket = list(_unpack_nelec(nelec))
    na_bra = math.comb(norb, nelec_bra[0])
    nb_bra = math.comb(norb, nelec_bra[1])
    na_ket = math.comb(norb, nelec_ket[0])
    nb_ket = math.comb(norb, nelec_ket[1])

    bravecs = np.empty((n_bra, na_bra, nb_bra))
    ketvecs = np.empty((n_ket, na_ket, nb_ket))
    for _nbra in range(n_bra): bravecs[_nbra] = np.random.random((na_bra, nb_bra))
    for _nket in range(n_ket): ketvecs[_nket] = np.random.random((na_ket, nb_ket))
    #for _nbra in range(n_bra): bravecs[_nbra] = np.arange(na_bra* nb_bra).reshape((na_bra, nb_bra)) + 0.5
    #for _nket in range(n_ket): ketvecs[_nket] = np.arange(na_ket* nb_ket).reshape((na_ket, nb_ket)) - 0.5
    
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
      tdm1s_c, tdm2s_c = multi_gpu_loop(bravecs, ketvecs,norb,nelec)
      tdm1s, tdm2s = o0_loop(bravecs, ketvecs,norb,nelec)
      tdm1s_correct = np.allclose(tdm1s, tdm1s_c)
      tdm2s_correct = np.allclose(tdm2s, tdm2s_c)
      if tdm1s_correct and tdm2s_correct: 
        print('TDM12s loop calculated correctly')
      else:
        print('TDM12s loop incorrect')
        print(tdm1s)
        print(tdm1s_c)
        print("tdm1 correct?", tdm1s_correct) 
        print("tdm2 correct?", tdm2s_correct) 
        exit()
    elif custom_fci and use_gpu and mgpu_fci: 
      print("in gpu channel")
      tdm1s, tdm12s = multi_gpu_loop(bravecs, ketvecs,norb,nelec)
    else:
      tdm1s, tdm2s = o0_loop(bravecs, ketvecs,norb,nelec)
    return tdm1s, tdm2s

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


  norb, nelec = 8, 7
  n_bra, n_ket = 5,3
  test_tdm12_loop(n_bra, n_ket,norb, nelec)
  #norb, nelec = 11, 15
  #n_bra, n_ket = 4,3
  #test_tdm12_loop(n_bra, n_ket,norb, nelec)
  if gpu_run: libgpu.destroy_device(gpu)

