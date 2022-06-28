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
from mrh.my_pyscf.grad import mspdft
from pyscf.fci import direct_spin1
from mrh.my_pyscf.grad.mspdft import _unpack_state

# TODO: state-average-mix generalization ?
def make_rdm1_heff_offdiag (mc, ci, si_bra, si_ket): 
    '''Compute <bra|O|ket> - sum_i <i|O|i>, where O is the 1-RDM
    operator product, and |bra> and |ket> are both states spanning the
    vector space of |i>, which are multi-determinantal many-electron
    states in an active space.

    Args:
        mc : object of class CASCI or CASSCF
            Only "ncas" and "nelecas" are used, to determine Hilbert
            of ci
        ci : ndarray or list of length (nroots)
            Contains CI vectors spanning a model space
        si_bra : ndarray of shape (nroots)
            Coefficients of ci elements for state |bra>
        si_ket : ndarray of shape (nroots)
            Coefficients of ci elements for state |ket>

    Returns:
        casdm1 : ndarray of shape [ncas,]*2
            Contains O = p'q case
    '''
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
    casdm1 -= np.tensordot (si_diag, ddm1, axes=1)
    return casdm1

def get_center_of_molecule(mol,center):
    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    # coords  = mol.atom_coords(unit='ANG')
    mass    = mol.atom_mass_list()
    if isinstance(center,str):
        if center.upper() == 'ORIGIN':
            coord = (0,0,0)
        elif center.upper() == 'MASS_CENTER':
            coord = np.einsum('i,ij->j', mass, coords) / mass.sum()
        elif center.upper() == 'NUC_CHARGE_CENTER':
            coord = np.einsum('z,zx->x', charges, coords) / charges.sum()
        else:
            raise RuntimeError ("Dipole's origin is not recognized")
    elif isinstance(center,str):
        coord = center
    else:
        raise RuntimeError ("Dipole's origin must be a string or tuple")
    print('coord',coord)
    return coord

def sipdft_HellmanFeynman_dipole (mc, si_bra=None, si_ket=None,
        state=None, mo_coeff=None, ci=None, si=None, center='Origin'):
    if state is None: state = mc.state
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci
    if si is None: si = mc.si
    if mc.frozen is not None:
        raise NotImplementedError
    t0 = (logger.process_clock (), logger.perf_counter ())

    ket, bra = _unpack_state (state)
    if si_bra is None: si_bra = mc.si[:,bra]
    if si_ket is None: si_ket = mc.si[:,ket]
    si_diag = si_bra * si_ket

    mol     = mc.mol
    ncore   = mc.ncore
    ncas    = mc.ncas
    nelecas = mc.nelecas
    nocc    = ncore + ncas
    mo_core = mo_coeff[:,:ncore]
    mo_cas  = mo_coeff[:,ncore:nocc]

    dm_core = np.dot(mo_core, mo_core.T) * 2

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
    dm_off = reduce(np.dot, (mo_cas, casdm1, mo_cas.T))

    dm = dm_diag + dm_off

    print('sipdft_HellmanFeynman_dipole',center)

    coord = get_center_of_molecule(mol,center)
    with mol.with_common_orig(coord):
        ao_dip = mol.intor_symmetric('int1e_r', comp=3)
    elec_term = -np.einsum('xij,ij->x', ao_dip, dm).real
    return elec_term

def nuclear_dipole(mc):
    '''Compute nuclear contribution to the dipole moment'''
    mol = mc.mol
    charges = mol.atom_charges()                           
    coords  = mol.atom_coords()
    nucl_term = np.einsum('i,ix->x', charges, coords)
    return nucl_term

class ElectricDipole (mspdft.Gradients):

    def kernel (self, state=None, mo=None, ci=None, si=None, _freeze_is=False,
        level_shift=None, unit='Debye', center='Origin', **kwargs):
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
        d2f = self.base.diabatizer (ci=ci)[2]
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
        cput1 = lib.logger.timer (self, 'Lagrange gradient multiplier solution', *cput0)

        print('kernel',center)
        ham_response = self.get_ham_response (center=center, **kwargs)

        LdotJnuc = self.get_LdotJnuc (Lvec, center=center, **kwargs)
        
        mol_dip = ham_response + LdotJnuc

        mol_dip = self.convert_dipole (ham_response, LdotJnuc, mol_dip, unit=unit)
        return mol_dip

    def convert_dipole (self, ham_response, LdotJnuc, mol_dip, unit='Debye'):
        i = self.state
        if unit.upper() == 'DEBYE':
            ham_response *= nist.AU2DEBYE
            LdotJnuc     *= nist.AU2DEBYE
            mol_dip      *= nist.AU2DEBYE
        log = lib.logger.new_logger(self, self.verbose)
        log.note('CMS-PDFT PDM <{}|mu|{}>          {:>10} {:>10} {:>10}'.format(i,i,'X','Y','Z'))
        log.note('Hamiltonian Contribution (%s) : %9.5f, %9.5f, %9.5f', unit, *ham_response)
        log.note('Lagrange Contribution    (%s) : %9.5f, %9.5f, %9.5f', unit, *LdotJnuc)
        log.note('Permanent Dipole Moment  (%s) : %9.5f, %9.5f, %9.5f', unit, *mol_dip)
        return mol_dip

    def get_ham_response (self, si_bra=None, si_ket=None, state=None, atmlst=None, verbose=None, mo=None,
                    ci=None, eris=None, si=None, center='Origin', **kwargs):
        if state is None: state = self.state
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if si is None: si = self.base.si
        ket, bra = _unpack_state (state)
        if si_bra is None: si_bra = si[:,bra]
        if si_ket is None: si_ket = si[:,ket]
        fcasscf = self.make_fcasscf (state)
        fcasscf.mo_coeff = mo
        fcasscf.ci = ci
        print('get_ham_response',center)
        elec_term = sipdft_HellmanFeynman_dipole (fcasscf, si_bra=si_bra, si_ket=si_ket,
         state=state, mo_coeff=mo, ci=ci, si=si, center=center)
        nucl_term = nuclear_dipole(fcasscf)
        total = nucl_term + elec_term
        return total

    def get_LdotJnuc (self, Lvec, atmlst=None, verbose=None, mo=None,
        ci=None, eris=None, center='Origin', **kwargs):
        if atmlst is None: atmlst = self.atmlst
        if verbose is None: verbose = self.verbose
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if eris is None and self.eris is None:
            eris = self.eris = self.base.ao2mo (mo)
        elif eris is None:
            eris = self.eris
        ncas = self.base.ncas
        nelecas = self.base.nelecas
        mc = self.base

        ngorb, nci, nis = self.ngorb, self.nci, self.nis
        Lvec_v = Lvec[:ngorb+nci]
        Lorb, Lci = self.unpack_uniq_var (Lvec_v)

        mo_coeff = mc.mo_coeff
        ci = mc.ci
    
        mol   = mc.mol
        ncore = mc.ncore
        ncas  = mc.ncas
        nocc  = ncore + ncas
        nelecas = mc.nelecas
    
        mo_core = mo_coeff[:,:ncore]
        mo_cas = mo_coeff[:,ncore:nocc]

        # Orbital part
        # MO coeff contracted against Lagrange multipliers
        moL_coeff = np.dot (mo_coeff, Lorb)
        moL_core = moL_coeff[:,:ncore]
        moL_cas = moL_coeff[:,ncore:nocc]
    
        casdm1 = mc.fcisolver.make_rdm1(ci, ncas, nelecas)
    
        dmL_core = np.dot(moL_core, mo_core.T) * 2
        dmL_cas = reduce(np.dot, (moL_cas, casdm1, mo_cas.T))
        dmL_core += dmL_core.T
        dmL_cas += dmL_cas.T

        # CI part 
        casdm1_transit, _ = mc.fcisolver.trans_rdm12 (Lci, ci, ncas, nelecas)
        casdm1_transit += casdm1_transit.transpose (1,0)

        dm_cas_transit = reduce(np.dot, (mo_cas, casdm1_transit, mo_cas.T))

        # Expansion coefficients are already in Lagrange multipliers
        dm = dmL_core + dmL_cas + dm_cas_transit
        print('get_LdotJnuc',center)
        coord = get_center_of_molecule(mol,center)
        with mol.with_common_orig(coord):
            ao_dip = mol.intor_symmetric('int1e_r', comp=3)
        mol_dip_L = -np.einsum('xij,ji->x', ao_dip, dm).real
        
        return mol_dip_L
