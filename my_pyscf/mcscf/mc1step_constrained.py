''' 
MRH: In which I attempt to modify the pyscf MC-SCF class(es) to allow arbitrary constraints on the active orbitals by messing with the gradients
'''

import numpy as np
import scipy as sp
from pyscf.lo.orth import orth_ao
from pyscf.scf import hf
from pyscf.mcscf import mc1step, addons
from pyscf.mcscf.mc1step import expmat
from pyscf.tools import molden
from functools import reduce
from mrh.my_pyscf.scf import hf_as
from mrh.util.basis import is_basis_orthonormal, are_bases_orthogonal, basis_olap, orthonormalize_a_basis
from mrh.util.la import is_matrix_eye

def orth_orb (orb, ovlp):
    ovlp_orb = reduce (np.dot, [orb.conjugate ().T, ovlp, orb])
    evals, evecs = sp.linalg.eigh (ovlp_orb)
    idx = evals>1e-10
    return np.dot (orb, evecs[:,idx]) / np.sqrt (evals[idx])

def rotate_orb_cc_wrapper (casscf, mo, fcivec, fcasdm1, fcasdm2, eris, x0_guess=None,
                  conv_tol_grad=1e-4, max_stepsize=None, verbose=None):
    ncore = casscf.ncore
    ncas = casscf.ncas
    ncasrot = casscf.ncasrot
    nocc = ncore + ncas

    # Test to make sure the orbitals never leave the proper space
    cas_ao = casscf.cas_ao
    err = np.linalg.norm (mo[~cas_ao,ncore:nocc])
    assert (abs (err) < 1e-10), err

    ovlp_ao = casscf._scf.get_ovlp ()    
    mo2casrot = reduce (np.dot, [mo.conjugate ().T, ovlp_ao, casscf.casrot_coeff])
    a2c = mo2casrot[ncore:nocc,:ncasrot]
    proj = np.dot (a2c.conjugate ().T, a2c)
    evals, evecs = sp.linalg.eigh (proj)
    assert (np.all (np.logical_or (np.isclose (evals, 1), np.isclose (evals, 0))))
    idx = evals.argsort ()[::-1]
    evals = evals[idx]
    evecs = evecs[:,idx]
    mo2casrot[:,:ncasrot] = np.dot (mo2casrot[:,:ncasrot], evecs)
    casscf._u_casrot = mo2casrot

    rota = mc1step.rotate_orb_cc (casscf, mo, fcivec, fcasdm1, fcasdm2, eris, 
            x0_guess=x0_guess, conv_tol_grad=conv_tol_grad, max_stepsize=max_stepsize, 
            verbose=verbose)
    fock_mo = reduce (np.dot, [mo.conjugate ().T,
        casscf.get_fock (mo_coeff=mo, ci=fcivec, eris=eris, casdm1=fcasdm1(), verbose=verbose),
        mo])
    for u_mo, g_orb, njk, r0 in rota:
        ''' This is not very efficient, because it doesn't really take effect until the last microcycle, but I don't know what else to do '''
        idx = np.zeros(mo.shape[0], dtype=np.bool_)
        idx[:ncore] = True
        idx[nocc:] = True
        idx2 = np.ix_(idx,idx)
        fock_mo1 = reduce (np.dot, [u_mo.conjugate ().T, fock_mo, u_mo])[idx2]
        evals, evecs = sp.linalg.eigh (fock_mo1)
        evecs = evecs[:,evals.argsort ()]
        evecs[:,np.diag(evecs)<0] *= -1
        u_fock = np.eye (u_mo.shape[0], dtype=u_mo.dtype)
        u_fock[idx2] = evecs
        u_mo = np.dot (u_mo, u_fock)
        yield u_mo, g_orb, njk, r0

def casci_scf_relaxation (envs):
    mc = envs['casscf']
    oldverbose = mc._scf.verbose
    mc._scf.verbose = 0
    mc._scf.build_frozen_from_mo (envs['mo'], mc.ncore, mc.ncas, envs['casdm1'], envs['casdm2'])
    mc._scf.diis = None
    dm0 = mc.make_rdm1 (mo_coeff=envs['mo'], ci=envs['fcivec'], ncas=mc.ncas, nelecas=mc.nelecas, ncore=mc.ncore)
    mc._scf.kernel (dm0)
    mo_change = reduce (np.dot, [envs['mo'].conjugate ().T, mc._scf.get_ovlp (), mc._scf.mo_coeff])
    mo_change = np.dot (mo_change, mo_change.conjugate ().T) - np.eye (mo_change.shape[0])
    print (np.linalg.norm (mo_change))
    envs['mo'] = mc._scf.mo_coeff
    mc._scf.verbose = oldverbose

class CASSCF(mc1step.CASSCF):
    '''MRH: In principle, to render certain orbitals orthogonal to certain other orbitals, all I should need to do is to project
    the initial guess and then mess with the gradients.  In this way I should be able to apply arbitrary constraints to the
    active orbitals, without affecting the cores and virtuals and without writing a whole Lagrangian optimizer.  Fingers crossed.
    I'm going to restrict myself to excluding entire atomic orbitals for now

    Extra attributes:

    cas_ao : ndarray, shape=(nao)
        boolean mask array for ao's allowed to contribute to cas

    casrot_coeff : ndarray, shape=(nao,nmo)
        orbital coefficients describing rotation space, the first ncasrot of which contain the active orbitals

    ncasrot : int
        number of orthonormal orbitals spanning cas_ao

    Parent class documentation follows:

    ''' + mc1step.CASSCF.__doc__

    def __init__(self, mf, ncas, nelecas, ncore=None, frozen=None, cas_ao=None):
        self.cas_ao = cas_ao
        self.casrot_coeff = None
        self.ncasrot = None
        assert (isinstance (mf, hf_as.RHF))
        mc1step.CASSCF.__init__(self, mf, ncas, nelecas, ncore, frozen)

    def kernel (self, mo_coeff=None, ci0=None, cas_ao=None, callback=None, _kern=mc1step.kernel):
        '''MRH: The only thing I need to do is to project the mo_coeffs, then pass along to the parent class member.
        The active orbitals need to be directly projected away from the inactive, and a reasonable
        selection of cores needs to be made.

        Extra kwargs:

        cas_ao : list of ints or boolean mask array of shape=(nao)
            aos allowed to contribute to cas

        Parent class documentation follows:

        ''' + mc1step.CASSCF.kernel.__doc__

        if mo_coeff is None:
            mo_coeff = self.mo_coeff

        if cas_ao is None:
            cas_ao = self.cas_ao

        self.cas_ao, self.casrot_coeff, self.ncasrot = self.build_casrot (cas_ao)

        mo_coeff = self.project_init_guess (mo_coeff)

        #molden.from_mo (self.mol, 'init.molden', mo_coeff, occ=self._scf.mo_occ)
        self.mo_coeff = mo_coeff

        return mc1step.CASSCF.kernel (self, mo_coeff=mo_coeff, ci0=ci0, callback=callback, _kern=_kern)

    def build_casrot (self, cas_ao=None):
        if cas_ao is None:
            cas_ao = self.cas_ao
        nao = self._scf.get_ovlp ().shape[1]
        x = np.zeros (nao, np.bool_)
        x[cas_ao] = True
        cas_ao = x
        nocas_ao = np.logical_not (x)
        ovlp_ao = self._scf.get_ovlp ()

        idx_mat = np.ix_(cas_ao,cas_ao)
        ovlp_cas_ao = ovlp_ao[idx_mat] 
        evals, evecs = sp.linalg.eigh (ovlp_cas_ao)
        idx_lindep = evals > 1e-12
        ncasrot = np.count_nonzero (idx_lindep)
        p_coeff = np.zeros ((nao, ncasrot))
        p_coeff[cas_ao,:] = evecs[:,idx_lindep] / np.sqrt (evals[idx_lindep])

        projector = reduce (np.dot, [p_coeff, p_coeff.conjugate ().T, ovlp_ao])
        projector = np.eye (nao) - projector
        projector = np.dot (ovlp_ao, projector)
        evals, evecs = sp.linalg.eigh (projector, ovlp_ao)
        assert (np.all (np.logical_or (np.isclose (evals, 1), np.isclose (evals, 0)))), "{0}".format (evals)
        idx = np.isclose (evals, 1)
        q_coeff = evecs[:,idx]

        casrot_coeff = np.append (p_coeff, q_coeff, axis=1)

        # Check orthonormality
        err_mat = np.eye (casrot_coeff.shape[1]) - reduce (np.dot, [casrot_coeff.conjugate ().T, ovlp_ao, casrot_coeff])
        err_norm = np.asarray ([np.linalg.norm (e) for e in (err_mat[:ncasrot,:ncasrot], 
                                                             err_mat[ncasrot:,:ncasrot],
                                                             err_mat[:ncasrot,ncasrot:],
                                                             err_mat[ncasrot:,ncasrot:])])

        assert (np.linalg.norm (err_norm) < 1e-10), "norm sectors = {0}".format (err_norm)

        return cas_ao, casrot_coeff, ncasrot

    def project_init_guess (self, mo_coeff=None, cas_ao=None, prev_mol=None):

        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if prev_mol is not None:
            mo_coeff = addons.project_init_guess (self, mo_coeff, prev_mol)
        if cas_ao is None:
            cas_ao = self.cas_ao
        else:
            self.cas_ao, self.casrot_coeff, self.ncasrot = self.build_casrot (cas_ao)
        ncas = self.ncas
        ncore = self.ncore
        nocc = self.mol.nelectron // 2
        dm0 = 2 * np.dot (mo_coeff[:,:nocc], mo_coeff[:,:nocc].conjugate ().T)
        self._scf.kernel (dm0)
        nocc = ncas + ncore

        ovlp_ao = self._scf.get_ovlp ()
        fock_ao = self._scf.get_fock ()
        ncasrot = self.ncasrot
        u_casrot = self.casrot_coeff[:,:ncasrot]
        assert (np.allclose (u_casrot[~cas_ao,:], 0))
        projector = np.dot (u_casrot, u_casrot.conjugate ().T)

        # Project active orbitals
        mo_coeff[:,ncore:nocc] = reduce (np.dot, [projector, ovlp_ao, mo_coeff[:,ncore:nocc]])
        assert (np.allclose (mo_coeff[~cas_ao,ncore:nocc], 0))
        mo_coeff[:,ncore:nocc] = orth_orb (mo_coeff[:,ncore:nocc], ovlp_ao)
        # Remove active component of core orbitals
        mo_coeff[:,:ncore] -= reduce (np.dot, [mo_coeff[:,ncore:nocc], mo_coeff[:,ncore:nocc].conjugate ().T, ovlp_ao, mo_coeff[:,:ncore]])
        mo_coeff[:,:ncore] = orth_orb (mo_coeff[:,:ncore], ovlp_ao)
        # Remove core-active component of virtual orbitals
        mo_coeff[:,nocc:] -= reduce (np.dot, [mo_coeff[:,:nocc], mo_coeff[:,:nocc].conjugate ().T, ovlp_ao, mo_coeff[:,nocc:]])
        mo_coeff[:,nocc:] = orth_orb (mo_coeff[:,nocc:], ovlp_ao)
        assert (np.allclose (mo_coeff[~cas_ao,ncore:nocc], 0))
        assert (is_basis_orthonormal (mo_coeff, ovlp_ao))

        # sort active orbitals by energy
        mo_energy = np.einsum ('ip,ij,jp->p', mo_coeff.conjugate (), fock_ao, mo_coeff.T)
        amo_energy = mo_energy[ncore:nocc]
        amo_coeff = mo_coeff[:,ncore:nocc]
        idx = amo_energy.argsort ()
        amo_energy = amo_energy[idx]
        amo_coeff = amo_coeff[:,idx]
        mo_energy[ncore:nocc] = amo_energy
        mo_coeff[:,ncore:nocc] = amo_coeff

        nelecb = self.mol.nelectron // 2
        neleca = nelecb + (self.mol.nelectron % 2)
        mo_occ = np.zeros (mo_coeff.shape[1])
        mo_occ[:neleca] += 1
        mo_occ[:nelecb] += 1
        mo_energy[:ncore] -= 100
        mo_energy[nocc:] += 100
        casdm1 = np.diag (mo_occ[ncore:nocc])
        self._scf.build_frozen_from_mo (mo_coeff, ncore, ncas)
        #molden.from_mo (self.mol, 'scf_before.molden', mo_coeff, occ=mo_occ, ene=mo_energy)
        self._scf.diis = None
        dm0 = hf.make_rdm1 (mo_coeff, mo_occ)
        self._scf.kernel (dm0)
        amo_ovlp = reduce (np.dot, [mo_coeff[:,ncore:nocc].conjugate ().T, ovlp_ao, self._scf.mo_coeff[:,ncore:nocc]])
        amo_ovlp = np.dot (amo_ovlp, amo_ovlp.conjugate ().T)
        err = np.trace (amo_ovlp) - ncas
        mo_energy = np.copy (self._scf.mo_energy)
        mo_energy[:ncore] -= 100
        mo_energy[nocc:] += 100
        #molden.from_mo (self.mol, 'scf_after.molden', self._scf.mo_coeff, occ=self._scf.mo_occ, ene=mo_energy)
        assert (abs (err) < 1e-10), "{0}".format (amo_ovlp)
        assert (np.allclose (self._scf.mo_coeff[~cas_ao,ncore:nocc], 0))

        return self._scf.mo_coeff

    rotate_orb_cc = rotate_orb_cc_wrapper

    def pack_uniq_var (self, rot):
        u = self._u_casrot
        uH = u.conjugate ().T
        nmo = rot.shape[0]
        ncas = self.ncas
        ncore = self.ncore
        nocc = ncore + ncas
        ncasrot = self.ncasrot
        #Active space
        rot = np.dot (rot[ncore:nocc,:], u)[:,ncas:ncasrot].ravel ()
        return rot 

    def unpack_uniq_var (self, rot):
        u = self._u_casrot
        uH = u.conjugate ().T
        nmo = self.casrot_coeff.shape[1]
        ncas = self.ncas
        ncore = self.ncore
        nocc = ncore + ncas
        nvirt = nmo - nocc
        ncasrot = self.ncasrot
        n1 = ncas * (ncasrot - ncas)
        # Active space
        rot = rot[:n1].reshape (ncas, ncasrot - ncas)
        mat = np.zeros ((nmo,nmo), dtype=u.dtype)
        mat[ncore:nocc,ncas:ncasrot] = rot
        mat[ncore:nocc,:] = np.dot (mat[ncore:nocc,:ncasrot], uH[:ncasrot,:])
        return mat - mat.T

