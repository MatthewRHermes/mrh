from pyscf.mcscf import newton_casscf
from pyscf.grad import rks as rks_grad
from pyscf.dft import gen_grid
from pyscf.lib import logger, pack_tril, current_memory, tag_array
from pyscf.grad import sacasscf
from pyscf.mcscf.casci import cas_natorb
from mrh.my_pyscf.mcpdft.otpd import get_ontop_pair_density, _grid_ao2mo
from mrh.my_pyscf.mcpdft.pdft_veff import _contract_vot_rho, _contract_ao_vao
from mrh.util.rdm import get_2CDM_from_2RDM
from functools import reduce
from scipy import linalg
import numpy as np
import time, gc
from pyscf.data import nist
from pyscf import lib
from mrh.my_pyscf.grad import mcpdft

def mcpdft_HellmanFeynman_dipole (mc, ot, veff1, veff2, mo_coeff=None, ci=None, atmlst=None, mf_grad=None, verbose=None, max_memory=None, auxbasis_response=False):
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci
    if mf_grad is None: mf_grad = mc._scf.nuc_grad_method()
    if mc.frozen is not None:
        raise NotImplementedError
    if max_memory is None: max_memory = mc.max_memory
    t0 = (time.process_time (), time.time ())

    mol = mc.mol
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas
    nelecas = mc.nelecas

    mo_core = mo_coeff[:,:ncore]
    mo_cas = mo_coeff[:,ncore:nocc]
 
    casdm1 = mc.fcisolver.make_rdm1(ci, ncas, nelecas)
 
    dm_core = np.dot(mo_core, mo_core.T) * 2
    dm_cas = reduce(np.dot, (mo_cas, casdm1, mo_cas.T))
    dm = dm_core + dm_cas
 
    with mol.with_common_orig((0,0,0)):
        ao_dip = mol.intor_symmetric('int1e_r', comp=3)
    el_dip = np.einsum('xij,ij->x', ao_dip, dm).real
 
    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    nucl_dip = np.einsum('i,ix->x', charges, coords)
    cas_dip = nucl_dip - el_dip

    return cas_dip

class ElectricDipole (mcpdft.Gradients):

    def kernel (self, level_shift=None, unit='Debye', **kwargs):
        ''' Cache the effective Hamiltonian terms so you don't have to calculate them twice '''
        state = kwargs['state'] if 'state' in kwargs else self.state
        if state is None:
            raise NotImplementedError ('Gradient of PDFT state-average energy')
        self.state = state # Not the best code hygiene maybe
        mo = kwargs['mo'] if 'mo' in kwargs else self.base.mo_coeff
        ci = kwargs['ci'] if 'ci' in kwargs else self.base.ci
        if isinstance (ci, np.ndarray): ci = [ci] # hack hack hack...
        kwargs['ci'] = ci
        kwargs['veff1'], kwargs['veff2'] = self.base.get_pdft_veff (mo, ci[state], incl_coul=True, paaa_only=True)
        cput0 = (time.process_time(), time.time())
        log = lib.logger.new_logger(self, self.verbose)
        if 'atmlst' in kwargs:
            self.atmlst = kwargs['atmlst']

        if self.verbose >= lib.logger.WARN:
            self.check_sanity()
        if self.verbose >= lib.logger.INFO:
            self.dump_flags()

        conv, Lvec, bvec, Aop, Adiag = self.solve_lagrange (level_shift=level_shift, **kwargs)
        self.debug_lagrange (Lvec, bvec, Aop, Adiag, **kwargs)
        #if not conv: raise RuntimeError ('Lagrange multiplier determination not converged!')
        cput1 = lib.logger.timer (self, 'Lagrange gradient multiplier solution', *cput0)

#        fcasscf = self.make_fcasscf (state)
#        fcasscf.mo_coeff = mo
#        fcasscf.ci = ci[state]
        ham_response = self.get_ham_response (**kwargs)
#        lib.logger.info(self, '--------------- %s gradient Hamiltonian response ---------------',
#                    self.base.__class__.__name__)
#        rhf_grad._write(self, self.mol, ham_response, self.atmlst)
#        lib.logger.info(self, '----------------------------------------------')
#        cput1 = lib.logger.timer (self, 'Lagrange gradient Hellmann-Feynman determination', *cput1)

        LdotJnuc = self.get_LdotJnuc (Lvec, **kwargs)
#        lib.logger.info(self, '--------------- %s gradient Lagrange response ---------------',
#                    self.base.__class__.__name__)
#        rhf_grad._write(self, self.mol, LdotJnuc, self.atmlst)
#        lib.logger.info(self, '----------------------------------------------')
#        cput1 = lib.logger.timer (self, 'Lagrange gradient Jacobian', *cput1)


        mol_dip = ham_response + LdotJnuc

        if unit.upper() == 'DEBYE':
            ham_response *= nist.AU2DEBYE
            mol_dip      *= nist.AU2DEBYE
            log.note('CASSCF  Dipole moment(X, Y, Z, Debye): %8.5f, %8.5f, %8.5f', *ham_response)
            log.note('MC-PDFT Dipole moment(X, Y, Z, Debye): %8.5f, %8.5f, %8.5f', *mol_dip)
        else:
            log.note('CASSCF  Dipole moment(X, Y, Z, A.U.): %8.5f, %8.5f, %8.5f', *ham_response)
            log.note('MC-PDFT Dipole moment(X, Y, Z, A.U.): %8.5f, %8.5f, %8.5f', *mol_dip)
        return mol_dip

    def get_ham_response (self, state=None, atmlst=None, verbose=None, mo=None, ci=None, eris=None, mf_grad=None, veff1=None, veff2=None, **kwargs):
        if state is None: state = self.state
        if atmlst is None: atmlst = self.atmlst
        if verbose is None: verbose = self.verbose
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if (veff1 is None) or (veff2 is None):
            assert (False), kwargs
            veff1, veff2 = self.base.get_pdft_veff (mo, ci[state], incl_coul=True, paaa_only=True)
        fcasscf = self.make_fcasscf (state)
        fcasscf.mo_coeff = mo
        fcasscf.ci = ci[state]
        return mcpdft_HellmanFeynman_dipole (fcasscf, self.base.otfnal, veff1, veff2, mo_coeff=mo, ci=ci[state], atmlst=atmlst, mf_grad=mf_grad, verbose=verbose)

    def get_LdotJnuc (self, Lvec, state=None, atmlst=None, verbose=None, mo=None, ci=None, eris=None, mf_grad=None, **kwargs):
        if state is None: state = self.state
        if atmlst is None: atmlst = self.atmlst
        if verbose is None: verbose = self.verbose
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci[state]
        if eris is None and self.eris is None:
            eris = self.eris = self.base.ao2mo (mo)
        elif eris is None:
            eris = self.eris
        ncas = self.base.ncas
        nelecas = self.base.nelecas
        if getattr(self.base.fcisolver, 'gen_linkstr', None):
            linkstr  = self.base.fcisolver.gen_linkstr(ncas, nelecas, False)
        else:
            linkstr  = None
        mc = self.base
     #   mc = self.make_fcasscf (state)
     #   fcasscf = self.make_fcasscf (state)
     #   fcasscf.mo_coeff = mo
     #   fcasscf.ci = ci[state]


        # Just sum the weights now... Lorb can be implicitly summed
        # Lci may be in the csf basis
        Lorb, Lci = self.unpack_uniq_var (Lvec)

        mo_coeff = mc.mo_coeff
        ci = mc.ci
    
        mol = mc.mol
        ncore = mc.ncore
        ncas = mc.ncas
        nocc = ncore + ncas
        nelecas = mc.nelecas
    
        mo_occ = mo_coeff[:,:nocc]
        mo_core = mo_coeff[:,:ncore]
        mo_cas = mo_coeff[:,ncore:nocc]
    

        # Orb part
        # MRH: new 'effective' MO coefficients including contraction from the Lagrange multipliers
        moL_coeff = np.dot (mo_coeff, Lorb)
        moL_core = moL_coeff[:,:ncore]
        moL_cas = moL_coeff[:,ncore:nocc]
    
        # MRH: these SHOULD be state-averaged! Use the actual sacasscf object!
        casdm1 = mc.fcisolver.make_rdm1(ci, ncas, nelecas)
    
        # MRH: new density matrix terms
        dmL_core = np.dot(moL_core, mo_core.T) * 2
        dmL_cas = reduce(np.dot, (moL_cas, casdm1, mo_cas.T))
        dmL_core += dmL_core.T
        dmL_cas += dmL_cas.T

        # MRH: TDMs + c.c. instead of RDMs; 06/30/2020: new interface in mcscf.addons makes this much more transparent
        casdm1_transit = mc.fcisolver.trans_rdm1 (Lci, ci, ncas, nelecas)
        casdm1_transit += casdm1_transit.transpose (1,0)

        dm_cas_transit = reduce(np.dot, (mo_cas, casdm1_transit, mo_cas.T))

        dm = dmL_core + dmL_cas + dm_cas_transit

        with mol.with_common_orig((0,0,0)):
            ao_dip = mol.intor_symmetric('int1e_r', comp=3)
        mol_dip_L = -np.einsum('xij,ji->x', ao_dip, dm).real

        return mol_dip_L 
