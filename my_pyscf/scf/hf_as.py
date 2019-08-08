# get_occ_activespace was copied and modified from pyscf.scf.hf.get_occ 04/22/2018
from pyscf.scf import hf
from pyscf import ao2mo
import numpy as np
import scipy as sp
from functools import reduce
from pyscf.scf.hf import energy_elec
from pyscf.lib import logger

def metaclass (mf):
    class HFmetaclass (mf.__class__):
        def __init__(self, my_mf):
            self.__dict__.update (my_mf.__dict__)
        eig = RHFas.eig
        get_occ = RHFas.get_occ
        get_fock = RHFas.get_fock
        get_grad = RHFas.get_grad
        energy_tot = RHFas.energy_tot   
        build_frozen_from_mo = RHFas.build_frozen_from_mo
        set_frozdm = RHFas.set_frozdm
        get_fo_coeff = RHFas.get_fo_coeff
        get_ufo_coeff = RHFas.get_ufo_coeff
    return HFmetaclass (mf)

def update_rdm12 (u, dm1, dm2):
    '''
    PySCF convention: density matrix indices are backwards?
    '''
    dm1 = np.einsum ('ai,ab,bj->ij', u, dm1, u.conjugate ())
    if dm2 is not None:
        dm2 = np.einsum ('abcd,ai->ibcd', dm2, u)
        dm2 = np.einsum ('ibcd,bj->ijcd', dm2, u.conjugate ())
        dm2 = np.einsum ('ijcd,ck->ijkd', dm2, u)
        dm2 = np.einsum ('ijkd,dl->ijkl', dm2, u.conjugate ())
    return dm1, dm2

def get_occ_activespace(mf, mo_energy=None, mo_coeff=None):
    nmo = mo_energy.size
    ncore = mf.ncore
    nfroz = mf.nfroz
    nocc = ncore + nfroz
    nvirt = nmo - nocc

    idx_ov = np.zeros (mo_energy.size, dtype=np.bool_)
    idx_ov[:ncore] = True
    idx_ov[nocc:] = True

    mo_energy_reduced = mo_energy[idx_ov]
    idx_sort = mo_energy_reduced.argsort ()
    idx_unsort = idx_sort.argsort ()

    mo_occ_reduced = np.zeros_like (mo_energy_reduced)
    mo_occ_reduced[:ncore] = 2
    mo_occ_reduced = mo_occ_reduced[idx_unsort]

    mo_occ = np.zeros (nmo)
    mo_occ[:ncore] = mo_occ_reduced[:ncore]
    mo_occ[nocc:] = mo_occ_reduced[ncore:]

    if mf._fo_occ is not None:
        mo_occ[ncore:nocc] = mf._fo_occ
    else:
        assert (mf.mol.nelectron % 2 == 0)
        mo_energy_reduced = mo_energy[~idx_ov]
        idx_sort = mo_energy_reduced.argsort ()
        idx_unsort = idx_sort.argsort ()

        nocc_froz = (mf.mol.nelectron // 2) - nocc
        mo_occ_reduced = np.zeros_like (mo_energy_reduced)
        mo_occ_reduced[:nocc_froz] = 2
        mo_occ_reduced = mo_occ_reduced[idx_unsort]

        mo_occ[ncore:nocc] = mo_occ_reduced

    e_sort = np.sort (mo_energy)
    nocc = mf.mol.nelectron // 2

    if mf.verbose >= logger.INFO and nocc < nmo:
        if e_sort[nocc-1]+1e-3 > e_sort[nocc]:
            logger.warn(mf, 'HOMO %.15g == LUMO %.15g',
                        e_sort[nocc-1], e_sort[nocc])
        else:
            logger.info(mf, '  HOMO = %.15g  LUMO = %.15g',
                        e_sort[nocc-1], e_sort[nocc])

    if mf.verbose >= logger.DEBUG:
        np.set_printoptions(threshold=nmo)
        logger.debug(mf, '  mo_energy =\n%s', mo_energy)
        np.set_printoptions(threshold=1000)
    return mo_occ

def energy_tot_activespace (mf, dm=None, h1e=None, vhf=None):
    if mf.frozdm1 is not None and mf.frozdm2 is not None:
        return hf.energy_tot (mf, dm, h1e, vhf) + mf._e2_froz - mf._e1_froz
    elif mf.frozdm2 is not None:
        fo = mf.get_fo_coeff ()
        foH = fo.conjugate ().T
        dm1 = reduce (np.dot, [foH, mf._fo_occ, fo])
        e1_froz = energy_elec (mf, dm1)[1]
        return hf.energy_tot (mf, dm, h1e, vhf) + mf._e2_froz - e1_froz
    else:
        return hf.energy_tot (mf, dm, h1e, vhf)

class RHFas(hf.RHF):
    '''
    MRH: A class to do HF in a subspace with some frozen density matrices that have an arbitrary density matrix

    Additional attributes:

    wo_coeff : ndarray, shape=(nao,nmo)
        orthonormal orbital coefficients whose ncore:ncore+nfroz columns are frozen and not allowed to change

    frozdm1 : ndarray, shape=(nfrox,nfroz)
        one-body density matrix in frozen-orbital basis. if supplied

    frozdm2 : ndarray, shape=(nfroz,nfroz,nfroz,nfroz)
        two-body density matrix in frozen-orbital basis

    Parent class documentation follows

    ''' + hf.RHF.__doc__

    def __init__(self, mol):
        self.wo_coeff = np.eye (mol.nao_nr ())
        self.ncore = mol.nelectron // 2
        self.nfroz = 0
        self.frozdm1 = None
        self.frozdm2 = None
        self._fo_occ = None
        self._e1_froz = None
        self._e2_froz = None
        hf.RHF.__init__(self, mol)

    def build_frozen_from_mo (self, mo_coeff, ncore, nfroz, frozdm1=None, frozdm2=None, eri_fo=None):

        # get natural fo's
        wo_coeff = np.copy (mo_coeff)
        self.wo_coeff = wo_coeff
        self.ncore = ncore
        self.nfroz = nfroz
        self._fo_occ = None
        self._e1_froz = None
        self._e2_froz = None
        if frozdm1 is not None or frozdm2 is not None:
            self.set_frozdm (frozdm1, frozdm2, eri_fo)

    def get_fo_coeff (self):
        return self.wo_coeff[:,self.ncore:self.ncore+self.nfroz]

    def get_ufo_coeff (self):
        idx = np.zeros (self.wo_coeff.shape[1], np.bool_)
        idx[:self.ncore] = True
        idx[self.ncore+self.nfroz:] = True
        return self.wo_coeff[:,idx]

    def set_frozdm (self, frozdm1=None, frozdm2=None, eri_fo=None):
        self.frozdm1 = frozdm1
        self.frozdm2 = frozdm2
        ncore = self.ncore
        nfroz = self.nfroz
        nocc = ncore + self.nfroz
        fo_coeff = self.get_fo_coeff ()
        if frozdm1 is not None:
            fo_occ, u_no = sp.linalg.eigh (frozdm1)
            frozdm1, frozdm2 = update_rdm12 (u_no, frozdm1, frozdm2)
            assert (np.allclose (fo_occ, np.diag (frozdm1))), "fo_occ = {0}\nfrozdm1 =\n{1}".format (fo_occ, frozdm1)
            self.wo_coeff[:,ncore:nocc] = np.dot (self.wo_coeff[:,ncore:nocc], u_no)
            self._fo_occ = fo_occ
            dm1 = reduce (np.dot, [fo_coeff, frozdm1, fo_coeff.conjugate ().T])
            self._e1_froz = energy_elec (self, dm1)[1]
        if frozdm2 is not None:
            if eri_fo is None:
                eri_fo = ao2mo.full(self.mol, fo_coeff, compact=False).reshape (nfroz, nfroz, nfroz, nfroz)
            else:
                eri_fo = ao2mo.restore (1, eri_fo, nfroz)
            self._e2_froz = 0.5 * np.tensordot (eri_fo, frozdm2, axes=4)

    def eig (self, h, s):
        ncore = self.ncore
        nfroz = self.nfroz
        nocc = ncore + nfroz
        nmo = self.wo_coeff.shape[1]
        nvirt = nmo - nocc
        fo_coeff = self.get_fo_coeff ()
        ufo_coeff = self.get_ufo_coeff ()

        h_proj = reduce (np.dot, [ufo_coeff.conjugate ().T, h, ufo_coeff])
        s_proj = reduce (np.dot, [ufo_coeff.conjugate ().T, s, ufo_coeff])

        mo_energy, mo_coeff_ov = self._eigh (h_proj, s_proj)
        idx = mo_energy.argsort ()
        mo_energy = mo_energy[idx]
        mo_coeff_ov = np.dot (ufo_coeff, mo_coeff_ov[:,idx])

        fo_energy = np.einsum ('ip,ij,jp->p', fo_coeff.conjugate (), h, fo_coeff)

        mo_energy = np.concatenate ([mo_energy[:ncore], fo_energy, mo_energy[ncore:]])
        mo_coeff = np.concatenate ([mo_coeff_ov[:,:ncore], fo_coeff, mo_coeff_ov[:,ncore:]], axis=1)

        return mo_energy, mo_coeff

    get_occ = get_occ_activespace
    energy_tot = energy_tot_activespace
        
    def get_grad (self, mo_coeff, mo_occ, fock=None):
        ncore = self.ncore
        nocc = ncore + self.nfroz
        nmo = mo_occ.size
        if fock is None:
            dm1 = self.make_rdm1 (mo_coeff, mo_occ)
            fock = self.get_hcore (self.mol) + self.get_veff (self.mol, dm1)
        idx = np.zeros (nmo, dtype=np.bool_)
        idx[:ncore] = True
        idx[nocc:] = True
        mo_occ_reduced = mo_occ[idx]
        mo_coeff_reduced = mo_coeff[:,idx]
        return hf.get_grad (mo_coeff_reduced, mo_occ_reduced, fock)

    def get_fock (mf, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
        diis_start_cycle=None, level_shift_factor=None, damp_factor=None):
        ''' For diis to work right, I think I have to zero out the fock matrix elements that couple the stuff I've frozen here'''
        if h1e is None: h1e = mf.get_hcore()
        if vhf is None: vhf = mf.get_veff(dm=dm)
        f = h1e + vhf
        ncore = mf.ncore  
        nocc = mf.ncore + mf.nfroz

        #if cycle >= 0 and diis is not None:
        # I still don't know how to fix the diis here...
            #f = reduce (np.dot, [mf.wo_coeff.conjugate ().T, f, mf.wo_coeff])
            #f[:ncore,ncore:nocc] = f[nocc:,ncore:nocc] = f[ncore:nocc,:ncore] = f[ncore:nocc,nocc:] = 0
            #f[ncore:nocc,ncore:nocc] = np.diag (np.diag (f[ncore:nocc,ncore:nocc]))
            #f = reduce (np.dot, [mf.wo_coeff, f, mf.wo_coeff.conjugate ()])
        
        return hf.get_fock (mf, h1e=h1e, s1e=s1e, vhf=vhf, dm=dm, 
                cycle=cycle, diis=diis, diis_start_cycle=diis_start_cycle,
                level_shift_factor=level_shift_factor, damp_factor=damp_factor)



