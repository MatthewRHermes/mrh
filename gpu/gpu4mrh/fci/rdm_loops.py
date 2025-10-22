from mrh.my_pyscf.gpu import libgpu
import pyscf
import numpy as np
#from gpu4mrh import patch_pyscf
from pyscf import lib
from pyscf.fci.addons import _unpack_nelec
from mrh.my_pyscf.fci import dummy
from itertools import product

def trans_rdm12s(tdm1s, tdm2s, bravecs, ketvecs, norb, nelec, linkstr, reorder=True):
  from mrh.my_pyscf.gpu import libgpu
  from pyscf.lib import param
  gpu=param.use_gpu
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

  libgpu.copy_tdm1_host_to_page(gpu, tdm1s, size_tdm1_full)  
  libgpu.copy_tdm2_host_to_page(gpu, tdm2s, size_tdm2_full)  
  
  return tdm1s, tdm2s
  
def trans_rdm13h(tdm1h, tdm3h, bravecs, ketvecs, norb, nelec, spin, linkstr, reorder=True, cre=False):
  from mrh.my_pyscf.gpu import libgpu
  from pyscf.lib import param
  gpu=param.use_gpu
  nelec = list (_unpack_nelec (nelec))
  if not cre:
    bravecs, ketvecs = ketvecs, bravecs
    nelec[spin] -= 1

  nelec_ket = _unpack_nelec (nelec)
  nelec_bra = [x for x in nelec]
  nelec_bra[spin] += 1
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
    libgpu.pull_tdm3hab_v2_host(gpu, j, i, n_bra, n_ket, norb, cre, spin, count)

  libgpu.copy_tdm1_host_to_page(gpu, tdm1h, size_tdm1h_full) 
  libgpu.copy_tdm2_host_to_page(gpu, tdm3h, size_tdm3h_full) 
  return tdm1h, tdm3h
  
def trans_sfddm1 (sfudm1, bravecs, ketvecs, norb, nelec, linkstr):
    nelec = list(_unpack_nelec (nelec))
    nelec[0] -= 1
    nelec[1] += 1
    #return trans_sfudm1 (sfudm1, ketvecs, bravecs, norb, nelec, linkstr).conj ().T #FIX THIS
    return trans_sfudm1 (sfudm1, ketvecs, bravecs, norb, nelec, linkstr).conj ().transpose(0,1,3,2) 

def trans_sfudm1(sfudm1, bravecs, ketvecs, norb, nelec, linkstr): 
  from mrh.my_pyscf.gpu import libgpu
  from pyscf.lib import param
  gpu=param.use_gpu
  nelec_ket = _unpack_nelec (nelec)
  nelec_bra = list (_unpack_nelec (nelec))
  nelec_bra[0] += 1
  nelec_bra[1] -= 1
  nelecd = [nelec_bra[0], nelec_ket[1]]
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
    libgpu.pull_tdm1_host(gpu, j, i, n_ket, n_bra, size_sfudm, 1, count)
  libgpu.copy_tdm1_host_to_page(gpu, sfudm1, size_sfudm_full) 
  return sfudm1 

def trans_hhdm(hhdm, bravecs, ketvecs, norb, nelec, spin, linkstr):
  nelec = list(_unpack_nelec (nelec))
  nelec[int (spin>1)] -= 1
  nelec[int (spin>0)] -= 1
  #return trans_ppdm (hhdm, ketvecs, bravecs, norb, nelec, spin, linkstr).conj ().T ##FIX THIS
  return trans_ppdm (hhdm, ketvecs, bravecs, norb, nelec, spin, linkstr).conj ().transpose(0,1,3,2) 

  
def trans_ppdm(ppdm, bravecs, ketvecs, norb, nelec, spin, linkstr):
  from mrh.my_pyscf.gpu import libgpu
  from pyscf.lib import param
  gpu=param.use_gpu
  s1 = int (spin>1)
  s2 = int (spin>0)
  ndum = 2 - (spin%2)
  nelec_ket = _unpack_nelec (nelec)
  nelec_bra = list (_unpack_nelec (nelec))
  nelec_bra[s1] += 1
  nelec_bra[s2] += 1
  occ_a, occ_b = int (spin<2), int (spin>0)
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
  libgpu.copy_tdm1_host_to_page(gpu, ppdm, size_hhdm_full) 
  return ppdm

