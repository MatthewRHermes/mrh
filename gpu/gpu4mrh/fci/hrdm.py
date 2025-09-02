import numpy
from pyscf.lib import params
from pyscf.fci import direct_spin1, rdm
from pyscf.fci.addons import _unpack_nelec
from mrh.my_pyscf.fci import dummy
def _trans_rdm13hs (cre, cibra, ciket, norb, nelec, spin=0, link_index=None, reorder=True):
    ''' Evaluate the one-half- and three-half-particle transition density matrices between ci
    vectors in different Hilbert spaces: <cibra|r'p'q|ciket> and <cibra|r'|ciket>, where |cibra>
    has the same number of orbitals but one additional electron of the same spin as r compared to
    |ciket>.

    Args:
        cre: logical
            True: creation sector, <cibra|r'|ciket> and <cibra|r'p'q|ciket>
            False: destruction sector, <cibra|r|ciket> and <cibra|p'qr|ciket>
        cibra: ndarray
            CI vector in (norb,nelec+1) Hilbert space
        ciket: ndarray
            CI vector in (norb,nelec) Hilbert space
        norb: integer
            Number of spatial orbitals 
        nelec: integer or sequence of length 2
            Number of electrons in the ket Hilbert space

    Kwargs:
        link_index: tuple of length 2 of "linkstr" type ndarray
            linkstr arrays for the nelec+1 electrons in norb+1 orbitals Hilbert space.
            See pyscf.fci.gen_linkstr_index for the shape of "linkstr".

    Returns:
        tdm1h: ndarray of shape (norb,)
            One-half-particle transition density matrix between cibra and ciket.
        (tdm3ha, tdm3hb): ndarrays of shape (norb,norb,norb,)
            Three-half-particle transition density matrix between cibra and ciket, spin-up and
            spin-down cases of the full electron. Returned in Mulliken order with the half-electron
            always first and the full electron always second:
            tdm3ha[r,p,q] = <cibra|r'p'q|ciket> or <cibra|p'qr|ciket>
    '''
    nelec = list (_unpack_nelec (nelec))
    if not cre:
      cibra, ciket = ciket, cibra
      nelec[spin] -= 1
    nelec_ket = _unpack_nelec (nelec)
    nelec_bra = [x for x in nelec]
    nelec_bra[spin] += 1
    linkstr = direct_spin1._unpack (norb+1, nelec_bra, link_index)
    errmsg = ("For the half-particle transition density matrix functions, the linkstr must "
              "be for nelec+1 electrons occupying norb+1 orbitals.")
    for i in range (2): assert (linkstr[i].shape[1]==(nelec_bra[i]*(norb-nelec_bra[i]+2))), errmsg
    ciket = dummy.add_orbital (ciket, norb, nelec_ket, occ_a=(1-spin), occ_b=spin)
    cibra = dummy.add_orbital (cibra, norb, nelec_bra, occ_a=0, occ_b=0)
    fn_par = ('FCItdm12kern_a', 'FCItdm12kern_b')[spin]
    fn_ab = 'FCItdm12kern_ab'
    gpu = param.use_gpu
    if param.custom_fci and param.gpu_debug:
      ### OLD KERNEL
      tdm1h, tdm3h_par = rdm.make_rdm12_spin1 (fn_par, cibra, ciket, norb+1, nelec_bra, link_index, 2)
      if reorder: tdm1h, tdm3h_par = rdm.reorder_rdm (tdm1h, tdm3h_par, inplace=True)
      if spin:
        tdm3ha = rdm.make_rdm12_spin1 (fn_ab, ciket, cibra, norb+1, nelec_bra, link_index, 0)[1]
        tdm3ha = tdm3ha.transpose (3,2,1,0)
        tdm3hb = tdm3h_par
      else:
        tdm3ha = tdm3h_par
        tdm3hb = rdm.make_rdm12_spin1 (fn_ab, cibra, ciket, norb+1, nelec_bra, link_index, 0)[1]
      ### new kernel
      tdm1h_c = numpy.empty((norb+1, norb+1))
      tdm3ha_c = numpy.empty((norb+1, norb+1, norb+1, norb+1))
      tdm3hb_c = numpy.empty((norb+1, norb+1, norb+1, norb+1))
      libgpu.init_tdm1h(gpu, norb+1)
      na, nlinka = link_indexa.shape[:2] 
      nb, nlinkb = link_indexb.shape[:2] 
      libgpu.push_link_indexab(gpu, na, nb, nlinka, nlinkb, link_indexa, link_indexb)
      libgpu.init_tdm3hab(gpu, norb+1)
      libgpu.push_ci(gpu, cibra, ciket, na, nb)
      libgpu.compute_tdm13h_spin(gpu, na, nb, nlinka, nlinkb, norb+1, spin) #TODO: write a better name
      libgpu.pull_tdm1(gpu, tdm1h, norb+1)
      libgpu.pull_tdm3hab(gpu, tdm3ha_c, tdm3hb_c, norb+1)
      if spin: 
        if reorder: tdm1h_c, tdm3hb_c = rdm.reorder_rdm(tdm1h_c, tdm3hb_c, inplace=True)
        tdm3ha_c = tdm3ha_c.transpose(3,2,1,0)
      else:
        if reorder: tdm1h_c, tdm3ha_c = rdm.reoder_rdm (tdm1h_c, tdm3ha_c, inplace=True)
      tdm1_correct = numpy.allclose(tdm1h, tdm1h_c)
      tdm3ha_correct = numpy.allclose(tdm3ha, tdm3ha_c)
      tdm3hb_correct = numpy.allclose(tdm3hb, tdm3hb_c)
      if tdm1_correct and tdm3ha_correct and tdm3hb_correct: 
        print('Trans RDM13hs calculated correctly')
      else:
        print('Some issues')
    elif param.custom_fci:
      pass
    else:
      tdm1h, tdm3h_par = rdm.make_rdm12_spin1 (fn_par, cibra, ciket, norb+1, nelec_bra, link_index, 2)
      if reorder: tdm1h, tdm3h_par = rdm.reorder_rdm (tdm1h, tdm3h_par, inplace=True)
      if spin:
        tdm3ha = rdm.make_rdm12_spin1 (fn_ab, ciket, cibra, norb+1, nelec_bra, link_index, 0)[1]
        tdm3ha = tdm3ha.transpose (3,2,1,0)
        tdm3hb = tdm3h_par
      else:
        tdm3ha = tdm3h_par
        tdm3hb = rdm.make_rdm12_spin1 (fn_ab, cibra, ciket, norb+1, nelec_bra, link_index, 0)[1]
    tdm1h = tdm1h[-1,:-1]
    tdm3ha = tdm3ha[:-1,-1,:-1,:-1]
    tdm3hb = tdm3hb[:-1,-1,:-1,:-1]
    if not cre: 
      tdm1h = tdm1h.conj ()
      tdm3ha = tdm3ha.conj ().transpose (0,2,1)
      tdm3hb = tdm3hb.conj ().transpose (0,2,1)
    return tdm1h, (tdm3ha, tdm3hb)

