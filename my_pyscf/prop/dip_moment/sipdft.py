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
from mrh.my_pyscf.grad import sipdft
from pyscf.fci import direct_spin1
from mrh.my_pyscf.grad import mcpdft as mcpdft_grad

def make_rdm12_heff_offdiag (mc, ci, si_bra, si_ket): 
    # TODO: state-average-mix generalization
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

def sipdft_heff_response (mc_grad, mo=None, ci=None,
        si_bra=None, si_ket=None, state=None, ham_si=None, 
        e_mcscf=None, eris=None):
    ''' Compute the orbital and intermediate-state rotation response 
        vector in the context of an SI-PDFT gradient calculation '''
    mc = mc_grad.base
    if mo is None: mo = mc_grad.mo_coeff
    if ci is None: ci = mc_grad.ci
    if state is None: state = mc_grad.state
    if si_bra is None: si_bra = mc.si[:,state]
    if si_ket is None: si_ket = mc.si[:,state]
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

def sipdft_heff_HellmanFeynman (mc_grad, atmlst=None, mo=None, ci=None, si=None,
        si_bra=None, si_ket=None, state=None, eris=None, mf_grad=None,
        verbose=None, **kwargs):
    mc = mc_grad.base
    if atmlst is None: atmlst = mc_grad.atmlst
    if mo is None: mo = mc.mo_coeff
    if ci is None: ci = mc.ci
    if si is None: si = getattr (mc, 'si', None)
    if state is None: state = mc_grad.state
    if si_bra is None: si_bra = si[:,state]
    if si_ket is None: si_ket = si[:,state]
    if eris is None: eris = mc.ao2mo (mo)
    if mf_grad is None: mf_grad = mc._scf.nuc_grad_method ()
    if verbose is None: verbose = mc_grad.verbose
    ncore, nroots = mc.ncore, mc_grad.nroots
    log = logger.new_logger (mc_grad, verbose)
    ci0 = np.zeros_like (ci[0])

    # CASSCF grad with effective RDMs
    t0 = (logger.process_clock (), logger.perf_counter ())    
    casdm1, casdm2 = make_rdm12_heff_offdiag (mc, ci, si_bra, si_ket)
    casdm1 = 0.5 * (casdm1 + casdm1.T)
    casdm2 = 0.5 * (casdm2 + casdm2.transpose (1,0,3,2))
    dm12 = lambda * args: (casdm1, casdm2)
    fcasscf = mc_grad.make_fcasscf (state=state,
        fcisolver_attr={'make_rdm12' : dm12})
    # TODO: DFeri functionality
    # Perhaps by patching fcasscf.nuc_grad_method?
    fcasscf_grad = fcasscf.nuc_grad_method ()
    #fcasscf_grad = casscf_grad.Gradients (fcasscf)
    de = fcasscf_grad.kernel (mo_coeff=mo, ci=ci0, atmlst=atmlst, verbose=0)

    # subtract nuc-nuc and core-core (patching out simplified gfock terms)
    moH = mo.conj ().T
    f0 = (moH @ mc.get_hcore () @ mo) + eris.vhf_c
    mo_energy = f0.diagonal ().copy ()
    mo_occ = np.zeros_like (mo_energy)
    mo_occ[:ncore] = 2.0
    f0 *= mo_occ[None,:]
    dme0 = lambda * args: mo @ ((f0+f0.T)*.5) @ moH
    with lib.temporary_env (mf_grad, make_rdm1e=dme0, verbose=0):
     with lib.temporary_env (mf_grad.base, mo_coeff=mo, mo_occ=mo_occ):
        # Second level there should become unnecessary in future, if anyone
        # ever gets around to cleaning up pyscf.df.grad.rhf & pyscf.grad.rhf
        dde = mf_grad.kernel (mo_coeff=mo, mo_energy=mo_energy, mo_occ=mo_occ,
            atmlst=atmlst)
    de -= dde
    log.debug ('SI-PDFT gradient off-diagonal H-F terms:\n{}'.format (de))
    log.timer ('SI-PDFT gradient off-diagonal H-F terms', *t0)
    return de

def get_sarotfns (obj):
    if obj.upper () == 'CMS':
        from mrh.my_pyscf.grad.cmspdft import sarot_response, sarot_grad
    else:
        raise RuntimeError ('SI-PDFT type not supported')
    return sarot_response, sarot_grad

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

def sipdft_HellmanFeynman_dipole (mc, state=None, mo_coeff=None, ci=None, si=None, atmlst=None, verbose=None, max_memory=None, auxbasis_response=False):
    if state is None: state = mc.state
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci
    if si is None: si = mc.si
    if mc.frozen is not None:
        raise NotImplementedError
    if max_memory is None: max_memory = mc.max_memory
    t0 = (logger.process_clock (), logger.perf_counter ())

    si_bra = si[:,state]
    si_ket = si[:,state]
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
                                                           
    # ---- Nuclear contribution -----
    charges = mol.atom_charges()                           
    coords  = mol.atom_coords()                            
    nucl_dip = np.einsum('i,ix->x', charges, coords)       
    cas_dip = nucl_dip - el_dip                            
    return cas_dip                                         

class ElectricDipole (sipdft.Gradients):

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
            log.note('CMS-PDFT Dipole moment(X, Y, Z, Debye): %8.5f, %8.5f, %8.5f', *mol_dip)
        else:
            log.note('Hellmann-Feynman Term(X, Y, Z, A.U.): %8.5f, %8.5f, %8.5f', *ham_response)
            log.note('Lagrange Contribution(X, Y, Z, A.U.): %8.5f, %8.5f, %8.5f', *LdotJnuc)
            log.note('CMS-PDFT Dipole moment(X, Y, Z, A.U.): %8.5f, %8.5f, %8.5f', *mol_dip)
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

    def get_LdotJnuc (self, Lvec, state=None, atmlst=None, verbose=None, mo=None, ci=None, si=None, eris=None, **kwargs):
        if state is None: state = self.state
        if atmlst is None: atmlst = self.atmlst
        if verbose is None: verbose = self.verbose
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if si is None: si = self.base.si
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

        si_bra = si[:,state]
        si_ket = si[:,state]
        si_diag = si_bra * si_ket

        ngorb, nci, nis = self.ngorb, self.nci, self.nis
        Lvec_v = Lvec[:ngorb+nci]
        Lorb, Lci = self.unpack_uniq_var (Lvec_v)

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

        # CI part 
        casdm1_transit, _ = mc.fcisolver.trans_rdm12 (Lci, ci, ncas, nelecas)
        casdm1_transit += casdm1_transit.transpose (1,0)

        dm_cas_transit = reduce(np.dot, (mo_cas, casdm1_transit, mo_cas.T))

        # AOL: Expansion coefficients are already accounted for in Lagrange multipliers
        dm = dmL_core + dmL_cas + dm_cas_transit

        #charges = mol.atom_charges()
        #coords = mol.atom_coords()
        #nuc_charge_center = numpy.einsum('z,zx->x', charges, coords) / charges.sum()
        #with mol.set_common_orig_(nuc_charge_center)
        with mol.with_common_orig((0,0,0)):
            ao_dip = mol.intor_symmetric('int1e_r', comp=3)
        mol_dip_L = -np.einsum('xij,ji->x', ao_dip, dm).real
        
        return mol_dip_L
