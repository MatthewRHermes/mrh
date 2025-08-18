#!/usr/bin/env python
import ctypes
import numpy
from pyscf import lib
from pyscf.fci import cistring
from pyscf.fci.addons import _unpack_nelec
DEBUG=True
librdm = cistring.libfci
def _make_rdm1_spin1(fname, cibra, ciket, norb, nelec, link_index=None, use_gpu=False, gpu=None):
    assert (cibra is not None and ciket is not None)
    if link_index is None:
        neleca, nelecb = _unpack_nelec(nelec)
        link_indexa = link_indexb = cistring.gen_linkstr_index(range(norb), neleca)
        if neleca != nelecb:
            link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    else:
        link_indexa, link_indexb = link_index
    #print(link_index)
    na,nlinka = link_indexa.shape[:2]
    nb,nlinkb = link_indexb.shape[:2]
    cibra = numpy.ascontiguousarray(cibra)
    ciket = numpy.ascontiguousarray(ciket)
    assert (cibra.size == na*nb), '{} {} {}'.format (cibra.size, na, nb)
    assert (ciket.size == na*nb), '{} {} {}'.format (ciket.size, na, nb)
    if use_gpu and DEBUG:
      from mrh.my_pyscf.gpu import libgpu
      rdm_cpu = numpy.empty((norb,norb))
      fn = getattr(librdm, fname)
      fn(rdm_cpu.ctypes.data_as(ctypes.c_void_p),
       cibra.ctypes.data_as(ctypes.c_void_p),
       ciket.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(norb),
       ctypes.c_int(na), ctypes.c_int(nb),
       ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
       link_indexa.ctypes.data_as(ctypes.c_void_p),
       link_indexb.ctypes.data_as(ctypes.c_void_p))
      print("finished cpu version",flush=True)
      rdm_gpu = numpy.empty((norb,norb))
      libgpu.init_tdm1(gpu, norb)
      libgpu.push_ci(gpu, cibra, ciket, na, nb)
      if fname == 'FCItrans_rdm1a': 
        libgpu.push_link_indexa(gpu, na, nlinka, link_indexa) #TODO: move up to direct spin1 or just generate on the fly
        libgpu.compute_trans_rdm1a(gpu, na, nb, nlinka, nlinkb, norb) #TODO: update name
      elif fname == 'FCItrans_rdm1b':
        libgpu.push_link_indexb(gpu, nb, nlinkb, link_indexb) #TODO: move up to direct_spin1 or just generate on the fly
        libgpu.compute_trans_rdm1b(gpu, na, nb, nlinka, nlinkb, norb) #TODO: update name
      libgpu.pull_tdm1(gpu, rdm_gpu, norb)
      if (numpy.allclose(rdm_cpu, rdm_gpu)):
        print("RDM's calculate correctly")
      else: 
        print("error in make_rdm1_spin1")
        print("rdm_cpu")
        print(rdm_cpu)
        print("rdm_gpu")
        print(rdm_gpu)
      return rdm_cpu.T
    elif use_gpu:  
      from mrh.my_pyscf.gpu import libgpu
      rdm_gpu = numpy.empty((norb,norb))
      libgpu.init_tdm(gpu, norb)
      libgpu.push_ci(gpu, cibra, ciket, na, nb)
      if fname == 'FCItrans_rdm1a': 
        libgpu.push_link_index(gpu, na, nlinka, link_indexa)
        libgpu.compute_trans_rdm1a(gpu, na, nb, nlinka, nlinkb, norb)
      elif fname == 'FCItrans_rdm1b':
        libgpu.push_link_index(gpu, nb, nlinkb, link_indexb)
        libgpu.compute_trans_rdm1b(gpu, na, nb, nlinka, nlinkb, norb)
      libgpu.pull_tdm1(gpu, rdm_gpu, norb)
      return rdm_gpu.T

    else:
      rdm1 = numpy.empty((norb,norb))
      fn = getattr(librdm, fname)
      fn(rdm1.ctypes.data_as(ctypes.c_void_p),
       cibra.ctypes.data_as(ctypes.c_void_p),
       ciket.ctypes.data_as(ctypes.c_void_p),
       ctypes.c_int(norb),
       ctypes.c_int(na), ctypes.c_int(nb),
       ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
       link_indexa.ctypes.data_as(ctypes.c_void_p),
       link_indexb.ctypes.data_as(ctypes.c_void_p))
      return rdm1.T

def _reorder_rdm(rdm1, rdm2, inplace=False):
    nmo = rdm1.shape[0]
    if not inplace:
        rdm2 = rdm2.copy()
    for k in range(nmo):
        rdm2[:,k,k,:] -= rdm1.T

    # Employing the particle permutation symmetry, average over two particles
    # to reduce numerical round off error
    rdm2 = lib.transpose_sum(rdm2.reshape(nmo*nmo,-1), inplace=True) * .5
    return rdm1, rdm2.reshape(nmo,nmo,nmo,nmo)

def _make_rdm12_spin1(fname, cibra, ciket, norb, nelec, link_index=None, symm=0, use_gpu=False, gpu=None):
    #FCItdm12_kern_a, FCItdm12kern_b, FCItdm12kern_ab
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
    if use_gpu and DEBUG: 
      from mrh.my_pyscf.gpu import libgpu
      rdm1_cpu = numpy.empty((norb,norb))
      rdm2_cpu = numpy.empty((norb,norb,norb,norb))
      rdm1_gpu = numpy.empty((norb,norb))
      rdm2_gpu = numpy.empty((norb,norb,norb,norb))
      librdm.FCIrdm12_drv(getattr(librdm, fname),
                        rdm1_cpu.ctypes.data_as(ctypes.c_void_p),
                        rdm2_cpu.ctypes.data_as(ctypes.c_void_p),
                        cibra.ctypes.data_as(ctypes.c_void_p),
                        ciket.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(norb),
                        ctypes.c_int(na), ctypes.c_int(nb),
                        ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
                        link_indexa.ctypes.data_as(ctypes.c_void_p),
                        link_indexb.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(symm))
      print("finished cpu version",flush=True)
      libgpu.init_tdm1(gpu, norb)
      libgpu.init_tdm2(gpu, norb)
      libgpu.push_ci(gpu, cibra, ciket, na, nb)
      libgpu.push_link_index_ab(gpu, na, nb, nlinka, nlinkb, link_indexa, link_indexb) #TODO: move this to direct_spin1 or generate on the fly
      if fname == 'FCItdm12kern_a': 
        libgpu.compute_tdm12kern_a(gpu, na, nb, nlinka, nlinkb, norb)
      if fname == 'FCItdm12kern_b': 
        libgpu.compute_tdm12kern_b(gpu, na, nb, nlinka, nlinkb, norb)
      if fname == 'FCItdm12kern_ab': 
        libgpu.compute_tdm12kern_ab(gpu, na, nb, nlinka, nlinkb, norb)
      libgpu.pull_tdm1(gpu, rdm1_gpu, norb)
      libgpu.pull_tdm2(gpu, rdm2_gpu, norb)
      rdm1_correct = numpy.allclose(rdm1_cpu, rdm1_gpu)
      rdm2_correct = numpy.allclose(rdm2_cpu, rdm2_gpu)
      if rdm1_correct and rdm2_correct:
        print("TDM12 calculated correctly")
      else: 
        print("Problem in TDM12")
        if rdm1_correct: print("TDM1 correct")
        else: 
          print("Incorrect TDM1")
          print("CPU TDM1")
          print(rdm1_cpu)
          print("GPU TDM1")
          print(rdm1_gpu)
        if rdm2_correct: print("TDM2 correct")
        else: 
          print("Incorrect TDM2")
          print("CPU TDM2")
          print(rdm2_cpu)
          #print(rdm2_cpu.reshape(norb*norb, norb*norb))
          print("GPU TDM2")
          print(rdm2_gpu)

      return rdm1_cpu.T, rdm2_cpu
    elif use_gpu: 
      from mrh.my_pyscf.gpu import libgpu
      rdm1_gpu = numpy.empty((norb,norb))
      rdm2_gpu = numpy.empty((norb,norb,norb,norb))
      libgpu.init_tdm1(gpu, norb)
      libgpu.init_tdm2(gpu, norb)
      libgpu.push_ci(gpu, cibra, ciket, na, nb)
      libgpu.push_link_index_ab(gpu, na, nb, nlinka, nlinkb, link_indexa, link_indexb) #TODO: move this to direct_spin1 because it's used with both a and b
      if fname == 'FCItdm12kern_a': 
        libgpu.compute_tdm12kern_a(gpu, norb, na, nb, nlinka, nlinkb)
      if fname == 'FCItdm12kern_b': 
        libgpu.compute_tdm12kern_b(gpu, norb, na, nb, nlinka, nlinkb)
      if fname == 'FCItdm12kern_ab': 
        libgpu.compute_tdm12kern_ab(gpu, norb, na, nb, nlinka, nlinkb)
      libgpu.pull_tdm1(gpu, rdm1_gpu, norb)
      libgpu.pull_tdm2(gpu, rdm2_gpu, norb)
      return rdm1_gpu.T, rdm2_gpu
    else: 
      rdm1 = numpy.empty((norb,norb))
      rdm2 = numpy.empty((norb,norb,norb,norb))
      librdm.FCIrdm12_drv(getattr(librdm, fname),
                        rdm1.ctypes.data_as(ctypes.c_void_p),
                        rdm2.ctypes.data_as(ctypes.c_void_p),
                        cibra.ctypes.data_as(ctypes.c_void_p),
                        ciket.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(norb),
                        ctypes.c_int(na), ctypes.c_int(nb),
                        ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
                        link_indexa.ctypes.data_as(ctypes.c_void_p),
                        link_indexb.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(symm))
      return rdm1.T, rdm2
