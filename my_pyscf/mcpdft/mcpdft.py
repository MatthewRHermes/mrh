import numpy as np
import time
from scipy import linalg
from pyscf import gto, dft, ao2mo, fci, mcscf, lib, __config__
from pyscf.lib import logger, temporary_env
from pyscf.mcscf import mc_ao2mo
from pyscf.mcscf.addons import StateAverageMCSCFSolver, state_average_mix, state_average_mix_
from mrh.my_pyscf.grad.mcpdft import Gradients
from mrh.my_pyscf.prop.dip_moment.mcpdft import ElectricDipole
from mrh.my_pyscf.mcpdft import pdft_veff, scf
from mrh.my_pyscf.mcpdft.otpd import get_ontop_pair_density
from mrh.my_pyscf.mcpdft.otfnal import otfnal, transfnal, ftransfnal
from mrh.util.rdm import get_2CDM_from_2RDM, get_2CDMs_from_2RDMs

def kernel (mc, ot, root=-1, verbose=None):
    ''' Calculate MC-PDFT total energy

        Args:
            mc : an instance of CASSCF or CASCI class
                Note: this function does not currently run the CASSCF or CASCI calculation itself
                prior to calculating the MC-PDFT energy. Call mc.kernel () before passing to this function!
            ot : an instance of on-top density functional class - see otfnal.py

        Kwargs:
            root : int
                If mc describes a state-averaged calculation, select the root (0-indexed)
                Negative number requests state-averaged MC-PDFT results (i.e., using state-averaged density matrices)
            verbose : int
                Verbosity of logger output; defaults to mc.verbose

        Returns:
            Total MC-PDFT energy including nuclear repulsion energy.
    '''
    if verbose is None: verbose = mc.verbose
    log = lib.logger.new_logger (mc, verbose)
    t0 = (time.clock (), time.time ())
    amo = mc.mo_coeff[:,mc.ncore:mc.ncore+mc.ncas]
    # make_rdm12s returns (a, b), (aa, ab, bb)

    mc_1root = mc
    if isinstance (mc, StateAverageMCSCFSolver) and root >= 0:
        mc_1root = mcscf.CASCI (mc._scf, mc.ncas, mc.nelecas)
        mc_1root.fcisolver = fci.solver (mc._scf.mol, singlet = False, symm = False)
        mc_1root.mo_coeff = mc.mo_coeff
        mc_1root.ci = mc.ci[root]
        mc_1root.e_tot = mc.e_states[root]
    dm1s = np.asarray (mc_1root.make_rdm1s ())
    adm1s = np.stack (mc_1root.fcisolver.make_rdm1s (mc_1root.ci, mc.ncas, mc.nelecas), axis=0)
    adm2 = get_2CDM_from_2RDM (mc_1root.fcisolver.make_rdm12 (mc_1root.ci, mc.ncas, mc.nelecas)[1], adm1s)
    spin = abs(mc.nelecas[0] - mc.nelecas[1])
    spin = abs(mc.nelecas[0] - mc.nelecas[1])
    omega, alpha, hyb = ot._numint.rsh_and_hybrid_coeff(ot.otxc, spin=spin)
    hyb_x, hyb_c = hyb
    if log.verbose >= logger.DEBUG or abs (hyb_x) > 1e-10 or abs (hyb_c) > 1e-10:
        adm2s = get_2CDMs_from_2RDMs (mc_1root.fcisolver.make_rdm12s (mc_1root.ci, mc.ncas, mc.nelecas)[1], adm1s)
        adm2s_ss = adm2s[0] + adm2s[2]
        adm2s_os = adm2s[1]
    t0 = logger.timer (log, 'rdms', *t0)

    Vnn = mc._scf.energy_nuc ()
    h = mc._scf.get_hcore ()
    dm1 = dm1s[0] + dm1s[1]
    if log.verbose >= logger.DEBUG or abs (hyb_x) > 1e-10:
        vj, vk = mc._scf.get_jk (dm=dm1s)
        vj = vj[0] + vj[1]
    else:
        vj = mc._scf.get_j (dm=dm1)
    Te_Vne = np.tensordot (h, dm1)
    # (vj_a + vj_b) * (dm_a + dm_b)
    E_j = np.tensordot (vj, dm1) / 2  
    # (vk_a * dm_a) + (vk_b * dm_b) Mind the difference!
    if log.verbose >= logger.DEBUG or abs (hyb_x) > 1e-10:
        E_x = -(np.tensordot (vk[0], dm1s[0]) + np.tensordot (vk[1], dm1s[1])) / 2
    else:
        E_x = 0
    logger.debug (log, 'CAS energy decomposition:')
    logger.debug (log, 'Vnn = %s', Vnn)
    logger.debug (log, 'Te + Vne = %s', Te_Vne)
    logger.debug (log, 'E_j = %s', E_j)
    logger.debug (log, 'E_x = %s', E_x)
    E_c = 0
    if log.verbose >= logger.DEBUG or abs (hyb_c) > 1e-10:
        # g_pqrs * l_pqrs / 2
        #if log.verbose >= logger.DEBUG:
        aeri = ao2mo.restore (1, mc.get_h2eff (mc.mo_coeff), mc.ncas)
        E_c = np.tensordot (aeri, adm2, axes=4) / 2
        E_c_ss = np.tensordot (aeri, adm2s_ss, axes=4) / 2
        E_c_os = np.tensordot (aeri, adm2s_os, axes=4) # ab + ba -> factor of 2
        logger.info (log, 'E_c = %s', E_c)
        logger.info (log, 'E_c (SS) = %s', E_c_ss)
        logger.info (log, 'E_c (OS) = %s', E_c_os)
        e_err = E_c_ss + E_c_os - E_c
        assert (abs (e_err) < 1e-8), e_err
        if isinstance (mc_1root.e_tot, float):
            e_err = mc_1root.e_tot - (Vnn + Te_Vne + E_j + E_x + E_c)
            assert (abs (e_err) < 1e-8), e_err
    if abs (hyb_x) > 1e-10 or abs (hyb_c) > 1e-10:
        logger.debug (log, 'Adding %s * %s CAS exchange, %s * %s CAS correlation to E_ot', hyb_x, E_x, hyb_c, E_c)
    t0 = logger.timer (log, 'Vnn, Te, Vne, E_j, E_x', *t0)

    E_ot = get_E_ot (ot, dm1s, adm2, amo, max_memory=mc.max_memory)
    t0 = logger.timer (log, 'E_ot', *t0)
    e_tot = Vnn + Te_Vne + E_j + (hyb_x * E_x) + (hyb_c * E_c) + E_ot
    logger.note (log, 'MC-PDFT E = %s, Eot(%s) = %s', e_tot, ot.otxc, E_ot)

    return e_tot, E_ot

def get_E_ot (ot, oneCDMs, twoCDM_amo, ao2amo, max_memory=2000, hermi=1):
    ''' E_MCPDFT = h_pq l_pq + 1/2 v_pqrs l_pq l_rs + E_ot[rho,Pi] 
        or, in other terms, 
        E_MCPDFT = T_KS[rho] + E_ext[rho] + E_coul[rho] + E_ot[rho, Pi]
                 = E_DFT[1rdm] - E_xc[rho] + E_ot[rho, Pi] 
        Args:
            ot : an instance of otfnal class
            oneCDMs : ndarray of shape (2, nao, nao)
                containing spin-separated one-body density matrices
            twoCDM_amo : ndarray of shape (ncas, ncas, ncas, ncas)
                containing spin-summed two-body cumulant density matrix in an active space
            ao2amo : ndarray of shape (nao, ncas)
                containing molecular orbital coefficients for active-space orbitals

        Kwargs:
            max_memory : int or float
                maximum cache size in MB
                default is 2000
            hermi : int
                1 if 1CDMs are assumed hermitian, 0 otherwise

        Returns : float
            The MC-PDFT on-top exchange-correlation energy

    '''
    ni, xctype, dens_deriv = ot._numint, ot.xctype, ot.dens_deriv
    norbs_ao = ao2amo.shape[0]

    E_ot = 0.0

    t0 = (time.clock (), time.time ())
    make_rho = tuple (ni._gen_rho_evaluator (ot.mol, oneCDMs[i,:,:], hermi) for i in range(2))
    for ao, mask, weight, coords in ni.block_loop (ot.mol, ot.grids, norbs_ao, dens_deriv, max_memory):
        rho = np.asarray ([m[0] (0, ao, mask, xctype) for m in make_rho])
        if ot.verbose > logger.DEBUG and dens_deriv > 0:
            for ideriv in range (1,4):
                rho_test  = np.einsum ('ijk,aj,ak->ia', oneCDMs, ao[ideriv], ao[0])
                rho_test += np.einsum ('ijk,ak,aj->ia', oneCDMs, ao[ideriv], ao[0])
                logger.debug (ot, "Spin-density derivatives, |PySCF-einsum| = %s", linalg.norm (rho[:,ideriv,:]-rho_test))
        t0 = logger.timer (ot, 'untransformed density', *t0)
        Pi = get_ontop_pair_density (ot, rho, ao, oneCDMs, twoCDM_amo, ao2amo, dens_deriv, mask) 
        t0 = logger.timer (ot, 'on-top pair density calculation', *t0) 
        E_ot += ot.get_E_ot (rho, Pi, weight)
        t0 = logger.timer (ot, 'on-top exchange-correlation energy calculation', *t0) 

    return E_ot

def get_energy_decomposition (mc, ot, mo_coeff=None, ci=None):
    ''' Compute a decomposition of the MC-PDFT energy into nuclear potential, core, Coulomb, exchange,
    and correlation terms. The exchange-correlation energy at the MC-SCF level is also returned.
    This is not meant to work with MC-PDFT methods that are already hybrids. Most return arguments
    are lists if mc is a state average instance. '''
    e_tot, e_ot, e_mcscf, e_cas, ci, mo_coeff = mc.kernel (mo=mo_coeff, ci=ci)[:6]
    if isinstance (mc, StateAverageMCSCFSolver):
        e_tot = mc.e_states
    e_nuc = mc._scf.energy_nuc ()
    h = mc.get_hcore ()
    xfnal, cfnal = ot.split_x_c ()
    if isinstance (mc, StateAverageMCSCFSolver):
        e_core = []
        e_coul = []
        e_otx = []
        e_otc = []
        e_wfnxc = []
        for ci_i, ei_mcscf in zip (ci, e_mcscf):
            row = _get_e_decomp (mc, ot, mo_coeff, ci_i, ei_mcscf, e_nuc, h, xfnal, cfnal)
            e_core.append  (row[0])
            e_coul.append  (row[1])
            e_otx.append   (row[2])
            e_otc.append   (row[3])
            e_wfnxc.append (row[4])
    else:
        e_core, e_coul, e_otx, e_otc, e_wfnxc = _get_e_decomp (mc, ot, mo_coeff, ci, e_mcscf, e_nuc, h, xfnal, cfnal)
    return e_nuc, e_core, e_coul, e_otx, e_otc, e_wfnxc

def _get_e_decomp (mc, ot, mo_coeff, ci, e_mcscf, e_nuc, h, xfnal, cfnal):
    mc_1root = mcscf.CASCI (mc._scf, mc.ncas, mc.nelecas)
    mc_1root.fcisolver = fci.solver (mc._scf.mol, singlet = False, symm = False)
    mc_1root.mo_coeff = mo_coeff
    mc_1root.ci = ci
    dm1s = np.stack (mc_1root.make_rdm1s (), axis=0)
    dm1 = dm1s[0] + dm1s[1]
    j = mc_1root._scf.get_j (dm=dm1)
    e_core = np.tensordot (h, dm1, axes=2)
    e_coul = np.tensordot (j, dm1, axes=2) / 2
    adm1s = np.stack (mc_1root.fcisolver.make_rdm1s (ci, mc.ncas, mc.nelecas), axis=0)
    adm2 = get_2CDM_from_2RDM (mc_1root.fcisolver.make_rdm12 (mc_1root.ci, mc.ncas, mc.nelecas)[1], adm1s)
    mo_cas = mo_coeff[:,mc.ncore:][:,:mc.ncas]
    e_otx = get_E_ot (xfnal, dm1s, adm2, mo_cas, max_memory=mc.max_memory)
    e_otc = get_E_ot (cfnal, dm1s, adm2, mo_cas, max_memory=mc.max_memory)
    e_wfnxc = e_mcscf - e_nuc - e_core - e_coul
    return e_core, e_coul, e_otx, e_otc, e_wfnxc

# This is clumsy and hacky and should be fixed in pyscf.mcscf.addons eventually rather than here
class StateAverageMCPDFTSolver:
    pass
def sapdft_grad_monkeypatch_(mc):
    if isinstance (mc, StateAverageMCPDFTSolver):
        return mc
    class StateAverageMCPDFT (mc.__class__, StateAverageMCPDFTSolver):
        def nuc_grad_method (self):
            return Gradients (self)
    mc.__class__ = StateAverageMCPDFT
    return mc

class _PDFT ():
    # Metaclass parent; unusable on its own

    def __init__(self, scf, ncas, nelecas, my_ot=None, grids_level=None, **kwargs):
        # Keep the same initialization pattern for backwards-compatibility. Use a separate intializer for the ot functional
        try:
            super().__init__(scf, ncas, nelecas)
        except TypeError as e:
            # I think this is the same DFCASSCF problem as with the DF-SACASSCF gradients earlier
            super().__init__()
        keys = set (('e_ot', 'e_mcscf', 'get_pdft_veff', 'e_states', 'otfnal', 'grids', 'max_cycle_fp', 'conv_tol_ci_fp'))
        self.max_cycle_fp = getattr (__config__, 'mcscf_mcpdft_max_cycle_fp', 50)
        self.conv_tol_ci_fp = getattr (__config__, 'mcscf_mcpdft_conv_tol_ci_fp', 1e-8)
        self._keys = set ((self.__dict__.keys ())).union (keys)
        if my_ot is not None:
            self._init_ot_grids (my_ot, grids_level=grids_level)

    def _init_ot_grids (self, my_ot, grids_level=None):
        if isinstance (my_ot, (str, np.string_)):
            ks = dft.RKS (self.mol)
            if my_ot[:1].upper () == 'T':
                ks.xc = my_ot[1:]
                self.otfnal = transfnal (ks)
            elif my_ot[:2].upper () == 'FT':
                ks.xc = my_ot[2:]
                self.otfnal = ftransfnal (ks)
            else:
                raise NotImplementedError (('On-top pair-density exchange-correlation functional names other than '
                    '"translated" (t) or "fully-translated" (ft). Nonstandard functionals can be specified by passing '
                    'an object of class otfnal in place of a string.'))
        else:
            self.otfnal = my_ot
        self.grids = self.otfnal.grids
        if grids_level is not None:
            self.grids.level = grids_level
            assert (self.grids.level == self.otfnal.grids.level)
        # Make sure verbose and stdout don't accidentally change (i.e., in scanner mode)
        self.otfnal.verbose = self.verbose
        self.otfnal.stdout = self.stdout
        
    def kernel (self, mo=None, ci=None, **kwargs):
        # Hafta reset the grids so that geometry optimization works!
        self._init_ot_grids (self.otfnal.otxc, grids_level=self.grids.level)
        self.e_mcscf, self.e_cas, self.ci, self.mo_coeff, self.mo_energy = super().kernel (mo, ci, **kwargs)
        if isinstance (self, StateAverageMCSCFSolver):
            epdft = [kernel (self, self.otfnal, root=ix) for ix in range (len (self.e_states))]
            self.e_mcscf = self.e_states
            self.fcisolver.e_states = [e_tot for e_tot, e_ot in epdft]
            self.e_ot = [e_ot for e_tot, e_ot in epdft]
            self.e_tot = np.dot (self.e_states, self.weights)
        else:
            self.e_tot, self.e_ot = kernel (self, self.otfnal)
        return self.e_tot, self.e_ot, self.e_mcscf, self.e_cas, self.ci, self.mo_coeff, self.mo_energy

    def dump_flags (self, verbose=None):
        super().dump_flags (verbose=verbose)
        log = logger.new_logger(self, verbose)
        log.info ('on-top pair density exchange-correlation functional: %s', self.otfnal.otxc)

    def get_pdft_veff (self, mo=None, ci=None, incl_coul=False, paaa_only=False):
        ''' Get the 1- and 2-body MC-PDFT effective potentials for a set of mos and ci vectors

            Kwargs:
                mo : ndarray of shape (nao,nmo)
                    A full set of molecular orbital coefficients. Taken from self if not provided
                ci : list or ndarray
                    CI vectors. Taken from self if not provided
                incl_coul : logical
                    If true, includes the Coulomb repulsion energy in the 1-body effective potential.
                    In practice they always appear together.
                paaa_only : logical
                    If true, only the paaa 2-body effective potential elements are evaluated; the rest of ppaa are filled with zeros.

            Returns:
                veff1 : ndarray of shape (nao, nao)
                    1-body effective potential in the AO basis
                    May include classical Coulomb potential term (see incl_coul kwarg)
                veff2 : pyscf.mcscf.mc_ao2mo._ERIS instance
                    Relevant 2-body effective potential in the MO basis
        ''' 
        t0 = (time.clock (), time.time ())
        if mo is None: mo = self.mo_coeff
        if ci is None: ci = self.ci
        # If ci is not a list and mc is a state-average solver, use a different fcisolver for make_rdm
        mc_1root = self
        if isinstance (self, StateAverageMCSCFSolver) and not isinstance (ci, list):
            mc_1root = mcscf.CASCI (self._scf, self.ncas, self.nelecas)
            mc_1root.fcisolver = fci.solver (self._scf.mol, singlet = False, symm = False)
            mc_1root.mo_coeff = mo
            mc_1root.ci = ci
            mc_1root.e_tot = self.e_tot
        dm1s = np.asarray (mc_1root.make_rdm1s (mo_coeff=mo, ci=ci))
        adm1s = np.stack (mc_1root.fcisolver.make_rdm1s (ci, self.ncas, self.nelecas), axis=0)
        adm2 = get_2CDM_from_2RDM (mc_1root.fcisolver.make_rdm12 (ci, self.ncas, self.nelecas)[1], adm1s)
        mo_cas = mo[:,self.ncore:][:,:self.ncas]
        pdft_veff1, pdft_veff2 = pdft_veff.kernel (self.otfnal, adm1s, adm2, mo, self.ncore, self.ncas, max_memory=self.max_memory, paaa_only=paaa_only)
        if self.verbose > logger.DEBUG:
            logger.debug (self, 'Warning: memory-intensive lazy kernel for pdft_veff initiated for '
                'testing purposes; reduce verbosity to decrease memory footprint')
            pdft_veff1_test, _pdft_veff2_test = pdft_veff.lazy_kernel (self.otfnal, dm1s, adm2, mo_cas)
            old_eri = self._scf._eri
            self._scf._eri = _pdft_veff2_test
            with temporary_env (self.mol, incore_anyway=True):
                pdft_veff2_test = mc_ao2mo._ERIS (self, mo, method='incore')
            self._scf._eri = old_eri
            err = linalg.norm (pdft_veff1 - pdft_veff1_test)
            logger.debug (self, 'veff1 error: {}'.format (err))
            err = linalg.norm (pdft_veff2.vhf_c - pdft_veff2_test.vhf_c)
            logger.debug (self, 'veff2.vhf_c error: {}'.format (err))
            err = linalg.norm (pdft_veff2.papa - pdft_veff2_test.papa)
            logger.debug (self, 'veff2.ppaa error: {}'.format (err))
            err = linalg.norm (pdft_veff2.papa - pdft_veff2_test.papa)
            logger.debug (self, 'veff2.papa error: {}'.format (err))
            err = linalg.norm (pdft_veff2.j_pc - pdft_veff2_test.j_pc)
            logger.debug (self, 'veff2.j_pc error: {}'.format (err))
            err = linalg.norm (pdft_veff2.k_pc - pdft_veff2_test.k_pc)
            logger.debug (self, 'veff2.k_pc error: {}'.format (err))
        
        if incl_coul:
            pdft_veff1 += self.get_jk (self.mol, dm1s[0] + dm1s[1])[0]
        logger.timer (self, 'get_pdft_veff', *t0)
        return pdft_veff1, pdft_veff2

    def nuc_grad_method (self):
        return Gradients (self)

    def dip_moment (self, unit='Debye'):
        dip_obj =  ElectricDipole(self) 
        mol_dipole = dip_obj.kernel ()
        return mol_dipole

    def get_energy_decomposition (self, mo_coeff=None, ci=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if ci is None: ci = self.ci
        return get_energy_decomposition (self, self.otfnal, mo_coeff=mo_coeff, ci=ci)

    def state_average (self, weights=(0.5,0.5)):
        # This is clumsy and hacky and should be fixed in pyscf.mcscf.addons eventually rather than here
        return sapdft_grad_monkeypatch_(super ().state_average (weights=weights))

    def state_average_(self, weights=(0.5,0.5)):
        # This is clumsy and hacky and should be fixed in pyscf.mcscf.addons eventually rather than here
        sapdft_grad_monkeypatch_(super ().state_average_(weights=weights))
        return self

    def state_average_mix (self, fcisolvers=None, weights=(0.5,0.5)):
        # This is clumsy and hacky and should be fixed in pyscf.mcscf.addons eventually rather than here
        return sapdft_grad_monkeypatch_(state_average_mix (self, fcisolvers, weights))

    def state_average_mix_(self, fcisolvers=None, weights=(0.5,0.5)):
        # This is clumsy and hacky and should be fixed in pyscf.mcscf.addons eventually rather than here
        sapdft_grad_monkeypatch_(state_average_mix_(self, fcisolvers, weights))
        return self

    @property
    def otxc (self):
        return self.otfnal.otxc

    @otxc.setter
    def otxc (self, x):
        self._init_ot_grids (x, grids_level=self.otfnal.grids.level)

def get_mcpdft_child_class (mc, ot, ci_min='ecas', **kwargs):

    # Inheritance magic
    class PDFT (_PDFT, mc.__class__):
        if ci_min.lower () == 'epdft':
            casci=scf.mc1step_casci

    pdft = PDFT (mc._scf, mc.ncas, mc.nelecas, my_ot=ot, **kwargs)
    _keys = pdft._keys.copy ()
    pdft.__dict__.update (mc.__dict__)
    pdft._keys = pdft._keys.union (_keys)
    return pdft


    



