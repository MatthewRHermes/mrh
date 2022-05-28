from pyscf.lib import logger
from pyscf.mcscf import mc1step, newton_casscf
from functools import reduce
import numpy as np
import time, gc
from pyscf.data import nist
from pyscf import lib
from mrh.my_pyscf.grad import mcpdft
from mrh.my_pyscf.grad import sipdft
from mrh.my_pyscf.prop.dip_moment import sipdft
from pyscf.fci import direct_spin1
from mrh.my_pyscf.grad import mcpdft as mcpdft_grad

def sipdft_heff_response (mc_grad, mo=None, ci=None,
        si_bra=None, si_ket=None, state=None, ham_si=None, 
        e_mcscf=None, eris=None):
    ''' Compute the orbital and intermediate-state rotation response 
        vector in the context of an SI-PDFT gradient calculation '''
    mc = mc_grad.base
    if mo is None: mo = mc_grad.mo_coeff
    if ci is None: ci = mc_grad.ci
    if state is None: state = mc_grad.state
    print('states are  ',state[0],state[1])
    if si_bra is None: si_bra = mc.si[:,state[0]]
    if si_ket is None: si_ket = mc.si[:,state[1]]
    if ham_si is None: ham_si = mc.ham_si
    if e_mcscf is None: e_mcscf = mc.e_mcscf
    if eris is None: eris = mc.ao2mo (mo)
    nroots, ncore = mc_grad.nroots, mc.ncore
    moH = mo.conj ().T

    # Orbital rotation (no all-core DM terms allowed!)
    # (Factor of 2 is convention difference between mc1step and newton_casscf)
    casdm1, casdm2 = make_rdm12_heff_offdiag (mc, ci, si_bra, si_ket)
    casdm1 = 0.5 * (casdm1 + casdm1.T)
    casdm2 = 0.5 * (casdm2 + casdm2.transpose (1,0,3,2))
    vnocore = eris.vhf_c.copy ()
    vnocore[:,:ncore] = -moH @ mc.get_hcore () @ mo[:,:ncore]
    with lib.temporary_env (eris, vhf_c=vnocore):
        g_orb = 2 * mc1step.gen_g_hop (mc, mo, 1, casdm1, casdm2, eris)[0]
    g_orb = mc.unpack_uniq_var (g_orb)

    # Intermediate state rotation (TODO: state-average-mix generalization)
    ham_is = ham_si.copy ()
    ham_is[np.diag_indices (nroots)] = e_mcscf
    braH = np.dot (si_bra, ham_is)
    Hket = np.dot (ham_is, si_ket)
    si2 = si_bra * si_ket
    g_is  = np.multiply.outer (si_ket, braH)
    g_is += np.multiply.outer (si_bra, Hket)
    g_is -= 2 * si2[:,None] * ham_is
    g_is -= g_is.T
    g_is = g_is[np.tril_indices (nroots, k=-1)]

    return g_orb, g_is

def make_rdm1_heff_offdiag (mc, ci, si_bra, si_ket): 
    # TODO: state-average-mix generalization
    ncas, nelecas = mc.ncas, mc.nelecas
    nroots = len (ci)
    ci_arr = np.asarray (ci)
    ci_bra = np.tensordot (si_bra, ci_arr, axes=1)
    ci_ket = np.tensordot (si_ket, ci_arr, axes=1)
    casdm1, _ = direct_spin1.trans_rdm12 (ci_bra, ci_ket, ncas, nelecas)
    ddm1 = np.zeros ((nroots, ncas, ncas), dtype=casdm1.dtype)
    for i in range (nroots):
        ddm1[i,...], _ = direct_spin1.make_rdm12 (ci[i], ncas, nelecas)
    si_diag = si_bra * si_ket
    a= np.tensordot (si_diag, ddm1, axes=1)
    casdm1 -= np.tensordot (si_diag, ddm1, axes=1)
    return casdm1

def make_rdm12_heff_offdiag (mc, ci, si_bra, si_ket): 
    # TODO: state-average-mix generalization
    #print('During Gradient, CI',ci)
    ncas, nelecas = mc.ncas, mc.nelecas
    nroots = len (ci)
    ci_arr = np.asarray (ci)
    ci_bra = np.tensordot (si_bra, ci_arr, axes=1)
    ci_ket = np.tensordot (si_ket, ci_arr, axes=1)
    casdm1, casdm2 = direct_spin1.trans_rdm12 (ci_bra, ci_ket, ncas, nelecas)
    ddm1 = np.zeros ((nroots, ncas, ncas), dtype=casdm1.dtype)
    ddm2 = np.zeros ((nroots, ncas, ncas, ncas, ncas), dtype=casdm1.dtype)
    for i in range (nroots):
        ddm1[i,...], ddm2[i,...] = direct_spin1.make_rdm12 (ci[i], ncas, nelecas)
    si_diag = si_bra * si_ket
    casdm1 -= np.tensordot (si_diag, ddm1, axes=1)
    casdm2 -= np.tensordot (si_diag, ddm2, axes=1)
    return casdm1, casdm2

def sipdft_HellmanFeynman_dipole (mc, state=None, mo_coeff=None, ci=None, si=None, atmlst=None, verbose=None, max_memory=None, auxbasis_response=False):
    if state is None: state = mc.state
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci
    if si is None: si = mc.si
    if mc.frozen is not None:
        raise NotImplementedError
    if max_memory is None: max_memory = mc.max_memory
    t0 = (logger.process_clock (), logger.perf_counter ())

    si_bra = si[:,state[0]]
    si_ket = si[:,state[1]]
    si_diag = si_bra * si_ket

    mol = mc.mol                                           
    ncore = mc.ncore                                       
    ncas = mc.ncas                                         
    nelecas = mc.nelecas                                   
    nocc = ncore + ncas                                    
                                                           
    mo_core = mo_coeff[:,:ncore]                           
    mo_cas  = mo_coeff[:,ncore:nocc]                        
                                                           
    dm_core = np.dot(mo_core, mo_core.T) * 2               

    # ----- Electronic contribution ------
    dm_diag=np.zeros_like(dm_core)
    # Diagonal part
    for i, (amp, c) in enumerate (zip (si_diag, ci)):
        if not amp: continue
        casdm1 = mc.fcisolver.make_rdm1(ci[i], ncas, nelecas)     
        dm_cas = reduce(np.dot, (mo_cas, casdm1, mo_cas.T))    
        dm_i = dm_cas + dm_core 
        dm_diag += amp * dm_i
        
    # Off-diagonal part
    casdm1 = make_rdm1_heff_offdiag (mc, ci, si_bra, si_ket)
#    casdm1 = 0.5 * (casdm1 + casdm1.T)
    dm_off = reduce(np.dot, (mo_cas, casdm1, mo_cas.T))    

    dm = dm_diag + dm_off                                             

    #charges = mol.atom_charges()
    #coords = mol.atom_coords()
    #nuc_charge_center = numpy.einsum('z,zx->x', charges, coords) / charges.sum()
    #with mol.set_common_orig_(nuc_charge_center)
    with mol.with_common_orig((0,0,0)):                    
        ao_dip = mol.intor_symmetric('int1e_r', comp=3)    
    el_dip = np.einsum('xij,ij->x', ao_dip, dm).real       
                                                           
    return el_dip                                         

class TransitionDipole (sipdft.ElectricDipole):

    def kernel (self, state=None, mo=None, ci=None, si=None, _freeze_is=False, level_shift=None, unit='Debye', **kwargs):
        ''' Cache the Hamiltonian and effective Hamiltonian terms, and pass
            around the IS hessian

            eris, veff1, veff2, and d2f should be available to all top-level
            functions: get_wfn_response, get_Aop_Adiag, get_ham_repsonse, and
            get_LdotJnuc
 
            freeze_is == True sets the is component of the response to zero
            for debugging purposes
        '''
        if state is None: state = self.state
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if si is None: si = self.base.si
        if isinstance (ci, np.ndarray): ci = [ci] # hack hack hack...
        if state is None:
            raise NotImplementedError ('Gradient of PDFT state-average energy')
        self.state = state # Not the best code hygiene maybe
        nroots = self.nroots
        veff1 = []
        veff2 = []
        d2f = self.base.sarot_objfn (ci=ci)[2]
        for ix in range (nroots):
            v1, v2 = self.base.get_pdft_veff (mo, ci, incl_coul=True,
                paaa_only=True, state=ix)
            veff1.append (v1)
            veff2.append (v2)
        kwargs['veff1'], kwargs['veff2'] = veff1, veff2
        kwargs['d2f'] = d2f 

        cput0 = (logger.process_clock(), logger.perf_counter())
        log = lib.logger.new_logger(self, self.verbose)
        if 'atmlst' in kwargs:
            self.atmlst = kwargs['atmlst']

        if self.verbose >= lib.logger.WARN:
            self.check_sanity()
        if self.verbose >= lib.logger.INFO:
            self.dump_flags()
        
        conv, Lvec, bvec, Aop, Adiag = self.solve_lagrange (level_shift=level_shift, **kwargs)
        #self.debug_lagrange (Lvec, bvec, Aop, Adiag, **kwargs)
        cput1 = lib.logger.timer (self, 'Lagrange gradient multiplier solution', *cput0)

        ham_response = self.get_ham_response (**kwargs)

        LdotJnuc = self.get_LdotJnuc (Lvec, **kwargs)
        
        mol_dip = ham_response + LdotJnuc

        if unit.upper() == 'DEBYE':
            ham_response *= nist.AU2DEBYE
            LdotJnuc     *= nist.AU2DEBYE
            mol_dip      *= nist.AU2DEBYE
            log.note('Hellmann-Feynman Term(X, Y, Z, Debye): %8.5f, %8.5f, %8.5f', *ham_response)
            log.note('Lagrange Contribution(X, Y, Z, Debye): %8.5f, %8.5f, %8.5f', *LdotJnuc)
            log.note('CMS-PDFT transition dipole moment between states %i and %i (X, Y, Z, Debye): %8.5f, %8.5f, %8.5f', state[0], state[1], *mol_dip)
        else:
            log.note('Hellmann-Feynman Term(X, Y, Z, A.U.): %8.5f, %8.5f, %8.5f', *ham_response)
            log.note('Lagrange Contribution(X, Y, Z, A.U.): %8.5f, %8.5f, %8.5f', *LdotJnuc)
            log.note('CMS-PDFT transition dipole moment between states %i and %i (X, Y, Z, A.U.): %8.5f, %8.5f, %8.5f', state[0], state[1], *mol_dip)
        return mol_dip

    def get_ham_response (self, state=None, atmlst=None, verbose=None, mo=None, ci=None, eris=None, si=None, **kwargs):
        mc = self.base
        if state is None: state = self.state
        if atmlst is None: atmlst = self.atmlst
        if verbose is None: verbose = self.verbose
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if si is None: si = self.base.si

        fcasscf = self.make_fcasscf (state)
        fcasscf.mo_coeff = mo
        fcasscf.ci = ci
        return sipdft_HellmanFeynman_dipole (fcasscf, state=state, mo_coeff=mo, ci=ci, si=si, atmlst=atmlst, verbose=verbose)

    def get_wfn_response (self, si_bra=None, si_ket=None, state=None, mo=None,
            ci=None, si=None, eris=None, veff1=None, veff2=None,
            _freeze_is=False, **kwargs):
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if si is None: si = self.base.si
        if state is None: state = self.state
        if si_bra is None: si_bra = si[:,state[0]]
        if si_ket is None: si_ket = si[:,state[1]]
        log = lib.logger.new_logger (self, self.verbose)
        si_diag = si_bra * si_ket
        nroots, ngorb, nci = self.nroots, self.ngorb, self.nci
        ptr_is = ngorb + nci

        # Diagonal: PDFT component
        nlag = self.nlag-self.nis
        g_all_pdft = np.zeros (nlag)
        for i, (amp, c, v1, v2) in enumerate (zip (si_diag, ci, veff1, veff2)):
            if not amp: continue
            g_i = mcpdft_grad.Gradients.get_wfn_response (self,
                state=i, mo=mo, ci=ci, veff1=v1, veff2=v2, nlag=nlag, **kwargs)
            g_all_pdft += amp * g_i
            if self.verbose >= lib.logger.DEBUG:
                g_orb, g_ci = self.unpack_uniq_var (g_i)
                g_ci, g_is = self._separate_is_component (g_ci, ci=ci, symm=0)
                log.debug ('g_is pdft state {} component:\n{} * {}'.format (i,
                    amp, g_is))

        # DEBUG
        g_orb_pdft, g_ci = self.unpack_uniq_var (g_all_pdft)
        g_ci, g_is_pdft = self._separate_is_component (g_ci, ci=ci, symm=0)

        # Off-diagonal: heff component
        g_orb_heff, g_is_heff = sipdft_heff_response (self, mo=mo, ci=ci,
            si_bra=si_bra, si_ket=si_ket, eris=eris)

        log.debug ('g_is pdft total component:\n{}'.format (g_is_pdft))
        log.debug ('g_is heff component:\n{}'.format (g_is_heff))

        # Combine
        g_orb = g_orb_pdft + g_orb_heff
        g_is = g_is_pdft + g_is_heff
        if _freeze_is: g_is[:] = 0.0
        g_all = self.pack_uniq_var (g_orb, g_ci, g_is)

        return g_all
