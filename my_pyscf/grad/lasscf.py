#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
CASSCF analytical nuclear gradients

Ref.
J. Comput. Chem., 5, 589

MRH: copied from pyscf.grad.casscf.py on 12/07/2019
Contains my modifications for SA-CASSCF gradients
1. Generalized Fock has nonzero i->a and u->a
2. Memory footprint for differentiated eris bugfix
'''

'''
SV: added and modified for casscf.py for lasscf nuclear gradients
Also check lasscf_sync_o0.py where a nuc_grad_method has been defined with a Gradients class!
'''

from functools import reduce
import numpy
from pyscf import gto
from pyscf import lib
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.grad import casci as casci_grad
from pyscf.grad import rhf as rhf_grad  # noqa
from pyscf.grad.mp2 import _shell_prange
from pyscf.mcscf.addons import StateAverageMCSCFSolver
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF

def grad_elec(las_grad, mo_coeff=None, ci=None, atmlst=None, verbose=None):
    las =las_grad.base
    print (las)
    if mo_coeff is None: mo_coeff = las.mo_coeff
    #if ci is None: ci = mc.ci
    #if mc.frozen is not None:
        #raise NotImplementedError

    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.new_logger(las_grad, verbose)
    mol = las_grad.mol
    ncore = las.ncore
    ncas = las.ncas
    nocc = ncore + ncas
    nelecas = las.nelecas
    nao, nmo = mo_coeff.shape
    nao_pair = nao * (nao+1) // 2

    # Necessary kludge because gfock isn't zero in occ-virt space in SA-CASSCf
    # Among many other potential applications!
    if hasattr (las, '_tag_gfock_ov_nonzero'):
        if las._tag_gfock_ov_nonzero:
            nocc = nmo

    mo_occ = mo_coeff[:,:nocc]
    mo_core = mo_coeff[:,:ncore]
    mo_cas = mo_coeff[:,ncore:nocc]
    #print ("SV nocc, ncas, ncore = ", nocc, ncas, ncore)
    #print ("mo_cas shape =", mo_cas.shape)
    #print ("SV entering grad_elec in lasscf")
    lasdm1 = las.make_casdm1()
    #print ("SV lasdm1 shape = ",lasdm1.shape)
    lasdm2 = las.make_casdm2()
    #casdm1, casdm2 = mc.fcisolver.make_rdm12(ci, ncas, nelecas)

# gfock = Generalized Fock, Adv. Chem. Phys., 69, 63
    dm_core = numpy.dot(mo_core, mo_core.T) * 2
    dm_cas = reduce(numpy.dot, (mo_cas, lasdm1, mo_cas.T))
    # MRH flag: this is one of my kludges
    # It would be better to just pass the ERIS object used in orbital optimization
    # But I am too lazy at the moment
    aapa = ao2mo.kernel(mol, (mo_cas, mo_cas, mo_occ, mo_cas), compact=False)
    aapa = aapa.reshape(ncas,ncas,nocc,ncas)
    vj, vk = las._scf.get_jk(mol, (dm_core, dm_cas))
    h1 = las.get_hcore()
    vhf_c = vj[0] - vk[0] * .5
    vhf_a = vj[1] - vk[1] * .5
    gfock = numpy.zeros ((nocc, nocc))
    gfock[:,:ncore] = reduce(numpy.dot, (mo_occ.T, h1 + vhf_c + vhf_a, mo_core)) * 2
    gfock[:,ncore:ncore+ncas] = reduce(numpy.dot, (mo_occ.T, h1 + vhf_c, mo_cas, lasdm1))
    gfock[:,ncore:ncore+ncas] += numpy.einsum('uviw,vuwt->it', aapa, lasdm2)
    dme0 = reduce(numpy.dot, (mo_occ, (gfock+gfock.T)*.5, mo_occ.T))
    aapa = vj = vk = vhf_c = vhf_a = h1 = gfock = None

    dm1 = dm_core + dm_cas
    vj, vk = las_grad.get_jk(mol, (dm_core, dm_cas))
    vhf1c, vhf1a = vj - vk * .5
    hcore_deriv = las_grad.hcore_generator(mol)
    s1 = las_grad.get_ovlp(mol)

    diag_idx = numpy.arange(nao)
    diag_idx = diag_idx * (diag_idx+1) // 2 + diag_idx
    lasdm2_cc = lasdm2 + lasdm2.transpose(0,1,3,2)
    dm2buf = ao2mo._ao2mo.nr_e2(lasdm2_cc.reshape(ncas**2,ncas**2), mo_cas.T,
                                (0, nao, 0, nao)).reshape(ncas**2,nao,nao)
    dm2buf = lib.pack_tril(dm2buf)
    dm2buf[:,diag_idx] *= .5
    dm2buf = dm2buf.reshape(ncas,ncas,nao_pair)
    lasdm2 = lasdm2_cc = None

    if atmlst is None:
        atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    de = numpy.zeros((len(atmlst),3))

    max_memory = las_grad.max_memory - lib.current_memory()[0]
    # MRH: this originally implied that the memory footprint would be max(p1-p0) * max(q1-q0) * nao_pair
    # In fact, that's the size of dm2_ao AND EACH COMPONENT of the differentiated eris
    # So the actual memory footprint is 4 times that!
    blksize = int(max_memory*.9e6/8 / (4*(aoslices[:,3]-aoslices[:,2]).max()*nao_pair))
    blksize = min(nao, max(2, blksize))

    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        h1ao = hcore_deriv(ia)
        de[k] += numpy.einsum('xij,ij->x', h1ao, dm1)
        de[k] -= numpy.einsum('xij,ij->x', s1[:,p0:p1], dme0[p0:p1]) * 2

        q1 = 0
        for b0, b1, nf in _shell_prange(mol, 0, mol.nbas, blksize):
            q0, q1 = q1, q1 + nf
            dm2_ao = lib.einsum('ijw,pi,qj->pqw', dm2buf, mo_cas[p0:p1], mo_cas[q0:q1])
            shls_slice = (shl0,shl1,b0,b1,0,mol.nbas,0,mol.nbas)
            eri1 = mol.intor('int2e_ip1', comp=3, aosym='s2kl',
                             shls_slice=shls_slice).reshape(3,p1-p0,nf,nao_pair)
            de[k] -= numpy.einsum('xijw,ijw->x', eri1, dm2_ao) * 2
            eri1 = None
        de[k] += numpy.einsum('xij,ij->x', vhf1c[:,p0:p1], dm1[p0:p1]) * 2
        de[k] += numpy.einsum('xij,ij->x', vhf1a[:,p0:p1], dm_core[p0:p1]) * 2

    log.timer('CASSCF nuclear gradients', *time0)
    return de

def as_scanner(mcscf_grad):
    '''Generating a nuclear gradients scanner/solver (for geometry optimizer).

    The returned solver is a function. This function requires one argument
    "mol" as input and returns energy and first order nuclear derivatives.

    The solver will automatically use the results of last calculation as the
    initial guess of the new calculation.  All parameters assigned in the
    nuc-grad object and SCF object (DIIS, conv_tol, max_memory etc) are
    automatically applied in the solver.

    Note scanner has side effects.  It may change many underlying objects
    (_scf, with_df, with_x2c, ...) during calculation.

    Examples:

    >>> from pyscf import gto, scf, mcscf
    >>> mol = gto.M(atom='N 0 0 0; N 0 0 1.1', verbose=0)
    >>> mc_grad_scanner = mcscf.CASSCF(scf.RHF(mol), 4, 4).nuc_grad_method().as_scanner()
    >>> etot, grad = mc_grad_scanner(gto.M(atom='N 0 0 0; N 0 0 1.1'))
    >>> etot, grad = mc_grad_scanner(gto.M(atom='N 0 0 0; N 0 0 1.5'))
    '''
    print ("SV Entering as_scanner in lasscf")
    if isinstance(mcscf_grad, lib.GradScanner):
        return mcscf_grad

    logger.info(mcscf_grad, 'Create scanner for %s', mcscf_grad.__class__)
    name = mcscf_grad.__class__.__name__ + LASSCF_GradScanner.__name_mixin__
    return lib.set_class(LASSCF_GradScanner(mcscf_grad),
                         (LASSCF_GradScanner, mcscf_grad.__class__), name)

class LASSCF_GradScanner(lib.GradScanner):
    def __init__(self, g):
        lib.GradScanner.__init__(self, g)

    def __call__(self, mol_or_geom, **kwargs):
        if isinstance(mol_or_geom, gto.MoleBase):
            assert mol_or_geom.__class__ == gto.Mole
            mol = mol_or_geom
        else:
            mol = self.mol.set_geom_(mol_or_geom, inplace=False)
        self.reset(mol)

        mc_scanner = self.base
        e_tot = mc_scanner(mol)
        if isinstance(mc_scanner, StateAverageMCSCFSolver):
            e_tot = mc_scanner.e_average

        de = self.kernel(**kwargs)
        return e_tot, de


class Gradients(casci_grad.Gradients):
    '''Non-relativistic restricted Hartree-Fock gradients'''
    print ("SV entering Gradients in LASSCF")
    grad_elec = grad_elec

    def kernel(self, mo_coeff=None, ci=None, atmlst=None, verbose=None):
        print ("SV entering kernel")
        log = logger.new_logger(self, verbose)
        if atmlst is None:
            atmlst = self.atmlst
        else:
            self.atmlst = atmlst

        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.verbose >= logger.INFO:
            self.dump_flags()

        de = self.grad_elec(mo_coeff, ci, atmlst, log)
        self.de = de = de + self.grad_nuc(atmlst=atmlst)
        if self.mol.symmetry:
            self.de = self.symmetrize(self.de, atmlst)
        self._finalize()
        return self.de
   
    def _finalize(self):
        if self.verbose >= logger.NOTE:
            logger.note(self, '--------------- %s gradients ---------------',
                        self.base.__class__.__name__)
            self._write(self.mol, self.de, self.atmlst)
            logger.note(self, '----------------------------------------------')

    as_scanner = as_scanner

    to_gpu = lib.to_gpu

#Grad = Gradients

#from pyscf import mcscf
#mcscf.mc1step.CASSCF.Gradients = lib.class_as_method(Gradients)                                                                                                                                             
