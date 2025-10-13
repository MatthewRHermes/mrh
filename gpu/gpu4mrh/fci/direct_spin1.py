#!/usr/bin/env python

'''
GPU accelerated version of specific functions of pyscf/fci/direct_spin1.py
'''

import sys
import ctypes
import numpy
import scipy.linalg
from pyscf import lib
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.fci import cistring
from pyscf.fci import rdm
from pyscf.fci import spin_op
from pyscf.fci import addons
from pyscf.fci.spin_op import contract_ss
from pyscf.fci.addons import _unpack_nelec, civec_spinless_repr
from pyscf import __config__

libfci = cistring.libfci

def _trans_rdm1s(cibra, ciket, norb, nelec, link_index=None):
    r'''Spin separated transition 1-particle density matrices.
    The return values include two density matrices: (alpha,alpha), (beta,beta).
    See also function :func:`make_rdm1s`

    1pdm[p,q] = :math:`\langle q^\dagger p \rangle`
    '''
    rdm1a = rdm.make_rdm1_spin1('FCItrans_rdm1a', cibra, ciket,
                                norb, nelec, link_index)
    rdm1b = rdm.make_rdm1_spin1('FCItrans_rdm1b', cibra, ciket,
                                norb, nelec, link_index)
    return rdm1a, rdm1b

def _trans_rdm12s(cibra, ciket, norb, nelec, link_index=None, reorder=True):
    r'''Spin separated 1- and 2-particle transition density matrices.
    The return values include two lists, a list of 1-particle transition
    density matrices and a list of 2-particle transition density matrices.
    The density matrices are:
    (alpha,alpha), (beta,beta) for 1-particle transition density matrices;
    (alpha,alpha,alpha,alpha), (alpha,alpha,beta,beta),
    (beta,beta,alpha,alpha), (beta,beta,beta,beta) for 2-particle transition
    density matrices.

    1pdm[p,q] = :math:`\langle q^\dagger p\rangle`;
    2pdm[p,q,r,s] = :math:`\langle p^\dagger r^\dagger s q\rangle`.
    '''
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

    if custom_fci and custom_debug and use_gpu:
      dm1a, dm1b, dm2aa, dm2ab, dm2ba, dm2bb = trans_rdm12s_o0(cibra, ciket, norb, nelec, link_index, reorder)
      dm1a_c, dm1b_c, dm2aa_c, dm2ab_c, dm2ba_c, dm2bb_c = trans_rdm12s_o1(cibra, ciket, norb, nelec, link_index, reorder)
      dm1a_correct = numpy.allclose(dm1a, dm1a_c)
      dm1b_correct = numpy.allclose(dm1b, dm1b_c)
      dm2aa_correct = numpy.allclose(dm2aa, dm2aa_c)
      dm2ab_correct = numpy.allclose(dm2ab, dm2ab_c)
      dm2ba_correct = numpy.allclose(dm2ba, dm2ba_c)
      dm2bb_correct = numpy.allclose(dm2bb, dm2bb_c)
      if dm1a_correct*dm1b_correct*dm2aa_correct*dm2ab_correct*dm2ba_correct*dm2bb_correct: 
        print("All DMs calculated correctly")
      else:
        print("dm1a_correct?", dm1a_correct) 
        print("dm1b_correct?", dm1b_correct) 
        print("dm2aa_correct?", dm2aa_correct) 
        print("dm2ab_correct?", dm2ab_correct) 
        print("dm2ba_correct?", dm2ba_correct) 
        print("dm2bb_correct?", dm2bb_correct) 
        exit()
    elif custom_fci and use_gpu: 
      dm1a, dm1b, dm2aa, dm2ab, dm2ba, dm2bb = trans_rdm12s_o1(cibra, ciket, norb, nelec, link_index, reorder)
    else: 
      dm1a, dm1b, dm2aa, dm2ab, dm2ba, dm2bb = trans_rdm12s_o0(cibra, ciket, norb, nelec, link_index, reorder)
    return (dm1a, dm1b), (dm2aa, dm2ab, dm2ba, dm2bb)


def trans_rdm12s_o0(cibra, ciket, norb, nelec, link_index=None, reorder=True):
    dm1a, dm2aa = rdm.make_rdm12_spin1('FCItdm12kern_a', cibra, ciket,
                                     norb, nelec, link_index, 2)
    dm1b, dm2bb = rdm.make_rdm12_spin1('FCItdm12kern_b', cibra, ciket,
                                     norb, nelec, link_index, 2)
    _, dm2ab = rdm.make_rdm12_spin1('FCItdm12kern_ab', cibra, ciket,
                                     norb, nelec, link_index, 0)
    _, dm2ba = rdm.make_rdm12_spin1('FCItdm12kern_ab', ciket, cibra,
                                     norb, nelec, link_index, 0)
    dm2ba = dm2ba.transpose(3,2,1,0)
    if reorder:
      dm1a, dm2aa = rdm.reorder_rdm(dm1a, dm2aa, inplace=True)
      dm1b, dm2bb = rdm.reorder_rdm(dm1b, dm2bb, inplace=True)
    return dm1a, dm1b, dm2aa, dm2ab, dm2ba, dm2bb

def trans_rdm12s_o1(cibra, ciket, norb, nelec, link_index=None, reorder=True):
    from pyscf.lib import param
    use_gpu = param.use_gpu
    gpu=param.use_gpu
    from mrh.my_pyscf.gpu import libgpu
    assert (cibra is not None and ciket is not None)
    cibra = numpy.asarray(cibra, order='C')
    ciket = numpy.asarray(ciket, order='C')
    if link_index is None:
      neleca, nelecb = _unpack_nelec(nelec)
      link_indexa = link_indexb = cistring.gen_linkstr_index(range(norb), neleca)
      if neleca != nelecb:
        link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    else:
      link_indexa, link_indexb = link_index
    na,nlinka = link_indexa.shape[:2]
    nb,nlinkb = link_indexb.shape[:2]
    assert (cibra.size == na*nb)
    assert (ciket.size == na*nb)
    libgpu.init_tdm1(gpu, norb)
    libgpu.init_tdm2(gpu, norb)
    libgpu.push_ciket(gpu, ciket, na, nb, 0)
    libgpu.push_cibra(gpu, cibra, na, nb, 0)
    libgpu.push_link_index_ab(gpu, na, nb, nlinka, nlinkb, link_indexa, link_indexb) #TODO: move this to direct_spin1 or generate on the fly
    #dm1a, dm2aa = rdm.make_rdm12_spin1('FCItdm12kern_a', cibra, ciket, norb, nelec, link_index, 2)
    libgpu.compute_tdm12kern_a_v2(gpu, na, nb, nlinka, nlinkb, norb, 0)
    if reorder: libgpu.reorder_rdm(gpu, norb, 0)
    dm1a = numpy.empty((norb, norb))
    dm2aa = numpy.empty((norb, norb, norb, norb))
    libgpu.pull_tdm1(gpu, dm1a, norb, 0)
    libgpu.pull_tdm2(gpu, dm2aa, norb, 0)
    #dm1b, dm2bb = rdm.make_rdm12_spin1('FCItdm12kern_b', cibra, ciket, norb, nelec, link_index, 2)
    libgpu.compute_tdm12kern_b_v2(gpu, na, nb, nlinka, nlinkb, norb, 0)
    if reorder: libgpu.reorder_rdm(gpu, norb, 0)
    dm1b = numpy.empty((norb, norb))
    dm2bb = numpy.empty((norb, norb, norb, norb))
    libgpu.pull_tdm1(gpu, dm1b, norb, 0)
    libgpu.pull_tdm2(gpu, dm2bb, norb, 0)
    #_, dm2ab = rdm.make_rdm12_spin1('FCItdm12kern_ab', cibra, ciket, norb, nelec, link_index, 0)
    libgpu.compute_tdm12kern_ab_v2(gpu, na, nb, nlinka, nlinkb, norb, 0)
    dm2ab = numpy.empty((norb, norb, norb, norb))
    libgpu.pull_tdm2(gpu, dm2ab, norb, 0)
    #_, dm2ba = rdm.make_rdm12_spin1('FCItdm12kern_ab', ciket, cibra, norb, nelec, link_index, 0)
    #libgpu.push_ci(gpu, ciket, cibra, na, nb) ## in a future version, figure out a better way to do this
    libgpu.push_ciket(gpu, cibra, na, nb, 0)
    libgpu.push_cibra(gpu, ciket, na, nb, 0)
    libgpu.compute_tdm12kern_ab_v2(gpu, na, nb, nlinka, nlinkb, norb, 0)
    dm2ba = numpy.empty((norb, norb, norb, norb))
    libgpu.pull_tdm2(gpu, dm2ba, norb, 0)
    dm2ba = dm2ba.transpose(3,2,1,0)
    return dm1a.T, dm1b.T, dm2aa, dm2ab, dm2ba, dm2bb


