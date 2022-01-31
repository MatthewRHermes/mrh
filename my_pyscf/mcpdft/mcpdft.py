import numpy as np
import time
from scipy import linalg
from pyscf import gto, dft, ao2mo, fci, mcscf, lib, __config__
from pyscf.lib import logger, temporary_env
from pyscf.fci import cistring
from pyscf.dft import gen_grid
from pyscf.mcscf import mc_ao2mo, mc1step
from pyscf.fci.direct_spin1 import _unpack_nelec
from pyscf.mcscf.addons import StateAverageMCSCFSolver, state_average_mix
from pyscf.mcscf.addons import state_average_mix_, StateAverageMixFCISolver
from mrh.my_pyscf.mcpdft import pdft_veff, ci_scf
from mrh.my_pyscf.mcpdft.otpd import get_ontop_pair_density
from mrh.my_pyscf.mcpdft.otfnal import otfnal, transfnal, ftransfnal
from mrh.util.rdm import get_2CDM_from_2RDM, get_2CDMs_from_2RDMs

# TODO: negative number = state-average API unittest

def energy_tot (mc, ot=None, mo_coeff=None, ci=None, root=-1, verbose=None):
    ''' Calculate MC-PDFT total energy

        Args:
            mc : an instance of CASSCF or CASCI class
                Note: this function does not currently run the CASSCF or
                CASCI calculation itself prior to calculating the
                MC-PDFT energy. Call mc.kernel () before passing to this
                function!

        Kwargs:
            ot : an instance of on-top functional class - see otfnal.py
            mo_coeff : ndarray of shape (nao, nmo)
                Molecular orbital coefficients
            ci : ndarray or list
                CI vector or vectors. Must be consistent with the nroots
                of mc.
            root : int
                If mc describes a state-averaged calculation, select the
                root (0-indexed). Negative number requests state-average
                MC-PDFT results (i.e., using state-averaged density
                matrices).
            verbose : int
                Verbosity of logger output; defaults to mc.verbose

        Returns:
            e_tot : float
                Total MC-PDFT energy including nuclear repulsion energy
            E_ot : float
                On-top (cf. exchange-correlation) energy
    '''
    if ot is None: ot = mc.otfnal
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci
    if verbose is None: verbose = mc.verbose
    t0 = (logger.process_clock (), logger.perf_counter ())

    # Allow MC-PDFT to be subclassed, and also allow this function to be
    # called without mc being an instance of MC-PDFT class

    if callable (getattr (mc, 'make_rdms_mcpdft', None)):
        dm_list = mc.make_rdms_mcpdft (ot=ot, mo_coeff=mo_coeff, ci=ci,
            state=root)
    else:
        dm_list = make_rdms_mcpdft (mc, ot=ot, mo_coeff=mo_coeff, ci=ci,
            state=root)
    t0 = logger.timer (ot, 'rdms', *t0)


    if callable (getattr (mc, 'energy_mcwfn', None)):
        e_mcwfn = mc.energy_mcwfn (ot=ot, mo_coeff=mo_coeff, dm_list=dm_list,
            verbose=verbose)
    else:
        e_mcwfn = energy_mcwfn (mc, ot=ot, mo_coeff=mo_coeff, dm_list=dm_list,
            verbose=verbose)
    t0 = logger.timer (ot, 'MC wfn energy', *t0)


    if callable (getattr (mc, 'energy_dft', None)):
        e_dft = mc.energy_dft (ot=ot, dm_list=dm_list)
    else:
        e_dft = energy_dft (mc, ot=ot, dm_list=dm_list)
    t0 = logger.timer (ot, 'E_ot', *t0)

    e_tot = e_mcwfn + e_dft
    return e_tot, e_dft

# Consistency with PySCF convention
kernel = energy_tot # backwards compatibility
def energy_elec (mc, *args, **kwargs):
    e_tot, E_ot = energy_tot (mc, *args, **kwargs)
    e_elec = e_tot - mc._scf.energy_nuc ()
    return e_elec, E_ot

def make_rdms_mcpdft (mc, ot=None, mo_coeff=None, ci=None, state=-1):
    ''' Build the necessary density matrices for an MC-PDFT calculation 

        Args:
            mc : an instance of CASSCF or CASCI class
                Note: this function does not currently run the CASSCF or
                CASCI calculation itself

        Kwargs:
            ot : an instance of on-top functional class - see otfnal.py
            mo_coeff : ndarray of shape (nao, nmo)
                Molecular orbital coefficients
            ci : ndarray or list
                CI vector or vectors. If a list of many CI vectors, mc
                must be a state-average object with the correct nroots
            state : integer
                Indexes the CI vector. If negative and if mc.fcisolver
                is a state-average object, state-averaged density
                matrices are returned.

        Returns:
            dm1s : ndarray of shape (2,nao,nao)
                Spin-separated 1-RDM
            adm : (adm1s, adm2s)
                adm1s : ndarray of shape (2,ncas,ncas)
                    Spin-separated 1-RDM for the active orbitals
                adm2s : 3 ndarrays of shape (ncas,ncas,ncas,ncas)
                    First ndarray is spin-summed casdm2
                    Second ndarray is casdm2_aa + casdm2_bb
                    Third ndarray is casdm2_ab
    '''
    if ci is None: ci = mc.ci
    if ot is None: ot = mc.otfnal
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    ncore, ncas, nelecas = mc.ncore, mc.ncas, mc.nelecas
    nocc = ncore + ncas

    # figure out the correct RDMs to build (SA or SS?)
    _casdms = mc.fcisolver
    if state >= 0:
        ci = ci[state]
        if isinstance (mc.fcisolver, StateAverageMixFCISolver):
            p0 = 0
            _casdms = None
            for s in mc.fcisolver.fcisolvers:
                p1 = p0 + s.nroots
                if p0 <= state and state < p1:
                    _casdms = s
                    nelecas = mc.fcisolver._get_nelec (s, nelecas)
                    break
                p0 = p1
            if _casdms is None:
                raise RuntimeError ("Can't find FCI solver for state", state)
        else:
            _casdms = fci.solver (mc._scf.mol, singlet=False, symm=False)

    # Make the rdms
    # make_rdm12s returns (a, b), (aa, ab, bb)
    mo_cas = mo_coeff[:,ncore:nocc]
    moH_cas = mo_cas.conj ().T
    mo_core = mo_coeff[:,:ncore]
    moH_core = mo_core.conj ().T
    adm1s = np.stack (_casdms.make_rdm1s (ci, ncas, nelecas), axis=0)
    adm2s = _casdms.make_rdm12s (ci, ncas, nelecas)[1]
    adm2s = get_2CDMs_from_2RDMs (adm2s, adm1s)
    adm2_ss = adm2s[0] + adm2s[2]
    adm2_os = adm2s[1]
    adm2 = adm2_ss + adm2_os + adm2_os.transpose (2,3,0,1)
    dm1s = np.dot (adm1s, moH_cas)
    dm1s = np.dot (mo_cas, dm1s).transpose (1,0,2)
    dm1s += np.dot (mo_core, moH_core)[None,:,:]
    return dm1s, (adm1s, (adm2, adm2_ss, adm2_os))

def energy_mcwfn (mc, ot=None, mo_coeff=None, ci=None, dm_list=None,
        verbose=None):
    ''' Compute the parts of the MC-PDFT energy arising from the wave
        function

        Args:
            mc : an instance of CASSCF or CASCI class
                Note: this function does not currently run the CASSCF or
                CASCI calculation itself prior to calculating the
                MC-PDFT energy. Call mc.kernel () before passing to thiswould
                function!

        Kwargs:
            ot : an instance of on-top functional class - see otfnal.py
            mo_coeff : ndarray of shape (nao, nmo)
                contains molecular orbital coefficients
            ci : list or ndarray
                contains ci vectors
            dm_list : (dm1s, adm2)
                return arguments of make_rdms_mcpdft

        Returns:
            e_mcwfn : float
                Energy from the multiconfigurational wave function:
                nuclear repulsion + 1e + coulomb
    '''

    if ot is None: ot = mc.otfnal
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci
    if verbose is None: verbose = mc.verbose
    if dm_list is None: dm_list = mc.make_rdms_mcpdft (ot=ot,
        mo_coeff=mo_coeff, ci=ci)
    log = logger.new_logger (mc, verbose=verbose)
    ncas, nelecas = mc.ncas, mc.nelecas
    dm1s, (adm1s, (adm2, adm2_ss, adm2_os)) = dm_list

    spin = abs(nelecas[0] - nelecas[1])
    spin = abs(nelecas[0] - nelecas[1])
    omega, alpha, hyb = ot._numint.rsh_and_hybrid_coeff(ot.otxc, spin=spin)
    hyb_x, hyb_c = hyb

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
        E_x = -(np.tensordot (vk[0], dm1s[0]) + np.tensordot (vk[1], dm1s[1]))
        E_x /= 2.0
    else:
        E_x = 0
    log.debug ('CAS energy decomposition:')
    log.debug ('Vnn = %s', Vnn)
    log.debug ('Te + Vne = %s', Te_Vne)
    log.debug ('E_j = %s', E_j)
    log.debug ('E_x = %s', E_x)
    E_c = 0
    if log.verbose >= logger.DEBUG or abs (hyb_c) > 1e-10:
        # g_pqrs * l_pqrs / 2
        #if log.verbose >= logger.DEBUG:
        aeri = ao2mo.restore (1, mc.get_h2eff (mo_coeff), mc.ncas)
        E_c = np.tensordot (aeri, adm2, axes=4) / 2
        E_c_ss = np.tensordot (aeri, adm2_ss, axes=4) / 2
        E_c_os = np.tensordot (aeri, adm2_os, axes=4) # ab + ba -> factor of 2
        log.info ('E_c = %s', E_c)
        log.info ('E_c (SS) = %s', E_c_ss)
        log.info ('E_c (OS) = %s', E_c_os)
        e_err = E_c_ss + E_c_os - E_c
        assert (abs (e_err) < 1e-8), e_err
    if abs (hyb_x) > 1e-10 or abs (hyb_c) > 1e-10:
        log.debug (('Adding %s * %s CAS exchange, %s * %s CAS correlation to '
                    'E_ot'), hyb_x, E_x, hyb_c, E_c)
    e_mcwfn = Vnn + Te_Vne + E_j + (hyb_x * E_x) + (hyb_c * E_c) 
    return e_mcwfn

def energy_dft (mc, ot=None, mo_coeff=None, dm_list=None, max_memory=None,
        hermi=1):
    ''' Wrap to get_E_ot for subclassing. '''
    if ot is None: ot = mc.otfnal
    if dm_list is None: dm_list = mc.make_rdms_mcpdft ()
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if max_memory is None: max_memory = mc.max_memory
    ncore, ncas = mc.ncore, mc.ncas
    nocc = ncore + ncas
    mo_cas = mo_coeff[:,ncore:nocc]
    dm1s, (adm1s, (adm2, adm2_ss, adm2_os)) = dm_list
    return get_E_ot (ot, dm1s, adm2, mo_cas, max_memory=max_memory,
        hermi=hermi)

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
                containing spin-summed two-body cumulant density matrix
                in an active space
            ao2amo : ndarray of shape (nao, ncas)
                containing molecular orbital coefficients for
                active-space orbitals

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

    t0 = (logger.process_clock (), logger.perf_counter ())
    make_rho = tuple (ni._gen_rho_evaluator (ot.mol, oneCDMs[i,:,:], hermi) for
        i in range(2))
    for ao, mask, weight, coords in ni.block_loop (ot.mol, ot.grids, norbs_ao,
            dens_deriv, max_memory):
        rho = np.asarray ([m[0] (0, ao, mask, xctype) for m in make_rho])
        if ot.verbose > logger.DEBUG and dens_deriv > 0:
            for ideriv in range (1,4):
                rho_test  = np.einsum ('ijk,aj,ak->ia', oneCDMs, ao[ideriv],
                    ao[0])
                rho_test += np.einsum ('ijk,ak,aj->ia', oneCDMs, ao[ideriv],
                    ao[0])
                logger.debug (ot, ("Spin-density derivatives, |PySCF-einsum| ="
                    " %s"), linalg.norm (rho[:,ideriv,:]-rho_test))
        t0 = logger.timer (ot, 'untransformed density', *t0)
        Pi = get_ontop_pair_density (ot, rho, ao, oneCDMs, twoCDM_amo, ao2amo,
            dens_deriv, mask) 
        t0 = logger.timer (ot, 'on-top pair density calculation', *t0) 
        E_ot += ot.get_E_ot (rho, Pi, weight)
        t0 = logger.timer (ot, 'on-top energy calculation', *t0) 

    return E_ot

# TODO: more detailed docstring + unittest
def get_energy_decomposition (mc, ot, mo_coeff=None, ci=None):
    ''' Compute a decomposition of the MC-PDFT energy into nuclear
    potential, core, Coulomb, exchange, and correlation terms. The
    exchange-correlation energy at the MC-SCF level is also returned.
    This is not meant to work with MC-PDFT methods that are already
    hybrids. Most return arguments are lists if mc is a state average
    instance. '''
    e_tot, e_ot, e_mcscf, e_cas, ci, mo_coeff = mc.kernel (mo=mo_coeff,
        ci=ci)[:6]
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
        nelec_root = [mc.nelecas,]* len (e_mcscf)
        if isinstance (mc.fcisolver, StateAverageMixFCISolver):
            nelec_root = []
            for s in mc.fcisolver.fcisolvers:
                ne_root_s = mc.fcisolver._get_nelec (s, mc.nelecas)
                nelec_root.extend ([ne_root_s,]*s.nroots)
        for ci_i, ei_mcscf, nelec in zip (ci, e_mcscf, nelec_root):
            row = _get_e_decomp (mc, ot, mo_coeff, ci_i, ei_mcscf, e_nuc, h,
                xfnal, cfnal, nelec)
            e_core.append  (row[0])
            e_coul.append  (row[1])
            e_otx.append   (row[2])
            e_otc.append   (row[3])
            e_wfnxc.append (row[4])
    else:
        e_core, e_coul, e_otx, e_otc, e_wfnxc = _get_e_decomp (mc, ot,
            mo_coeff, ci, e_mcscf, e_nuc, h, xfnal, cfnal, mc.nelecas)
    return e_nuc, e_core, e_coul, e_otx, e_otc, e_wfnxc

def _get_e_decomp (mc, ot, mo_coeff, ci, e_mcscf, e_nuc, h, xfnal, cfnal,
        nelecas):
    ncore, ncas = mc.ncore, mc.ncas
    _rdms = mcscf.CASCI (mc._scf, ncas, nelecas)
    _rdms.fcisolver = fci.solver (mc._scf.mol, singlet = False, symm = False)
    _rdms.mo_coeff = mo_coeff
    _rdms.ci = ci
    _casdms = _rdms.fcisolver
    dm1s = np.stack (_rdms.make_rdm1s (), axis=0)
    dm1 = dm1s[0] + dm1s[1]
    j = _rdms._scf.get_j (dm=dm1)
    e_core = np.tensordot (h, dm1, axes=2)
    e_coul = np.tensordot (j, dm1, axes=2) / 2
    adm1s = np.stack (_casdms.make_rdm1s (ci, ncas, nelecas), axis=0)
    adm2 = get_2CDM_from_2RDM (_casdms.make_rdm12 (_rdms.ci, ncas, nelecas)[1],
        adm1s)
    mo_cas = mo_coeff[:,ncore:][:,:ncas]
    e_otx = get_E_ot (xfnal, dm1s, adm2, mo_cas, max_memory=mc.max_memory)
    e_otc = get_E_ot (cfnal, dm1s, adm2, mo_cas, max_memory=mc.max_memory)
    e_wfnxc = e_mcscf - e_nuc - e_core - e_coul
    return e_core, e_coul, e_otx, e_otc, e_wfnxc

# TODO: edit pyscf.mcscf.addons to look for a hook to a child-class gradients
# method, so that all of this "monkeypatch" nonsense can be deleted
class StateAverageMCPDFTSolver:
    pass
def sapdft_grad_monkeypatch_(mc):
    if isinstance (mc, StateAverageMCPDFTSolver):
        return mc
    class StateAverageMCPDFT (mc.__class__, StateAverageMCPDFTSolver):
        def nuc_grad_method (self):
            from mrh.my_pyscf.grad.mcpdft import Gradients
            return Gradients (self)
    mc.__class__ = StateAverageMCPDFT
    return mc

class _PDFT ():
    # Metaclass parent; unusable on its own

    def __init__(self, scf, ncas, nelecas, my_ot=None, grids_level=None,
            grids_attr={}, **kwargs):
        # Keep the same initialization pattern for backwards-compatibility.
        # Use a separate intializer for the ot functional
        try:
            super().__init__(scf, ncas, nelecas)
        except TypeError as e:
            # I think this is the same DFCASSCF problem as with the DF-SACASSCF
            # gradients earlier
            super().__init__()
        keys = set (('e_ot', 'e_mcscf', 'get_pdft_veff', 'e_states', 'otfnal',
            'grids', 'max_cycle_fp', 'conv_tol_ci_fp', 'mcscf_kernel'))
        self.max_cycle_fp = getattr (__config__, 'mcscf_mcpdft_max_cycle_fp',
            50)
        self.conv_tol_ci_fp = getattr (__config__,
            'mcscf_mcpdft_conv_tol_ci_fp', 1e-8)
        self.mcscf_kernel = super().kernel
        self._keys = set ((self.__dict__.keys ())).union (keys)
        if grids_level is not None:
            grids_attr['level'] = grids_level
        if my_ot is not None:
            self._init_ot_grids (my_ot, grids_attr=grids_attr)

    def _init_ot_grids (self, my_ot, grids_attr={}):
        old_grids = getattr (self, 'grids', None)
        if isinstance (my_ot, (str, np.string_)):
            ks = dft.RKS (self.mol)
            if my_ot[:1].upper () == 'T':
                ks.xc = my_ot[1:]
                self.otfnal = transfnal (ks)
            elif my_ot[:2].upper () == 'FT':
                ks.xc = my_ot[2:]
                self.otfnal = ftransfnal (ks)
            else:
                raise NotImplementedError (('On-top pair-density functional '
                    'names other than "translated" (t) or "fully-translated" '
                    '(ft). Nonstandard functionals can be specified by passing'
                    ' an object of class otfnal in place of a string.'))
        else:
            self.otfnal = my_ot
        if isinstance (old_grids, gen_grid.Grids):
            self.otfnal.grids = old_grids
        self.grids = self.otfnal.grids
        self.grids.__dict__.update (grids_attr)
        for key in grids_attr:
            assert (getattr (self.grids, key, None) == getattr (
                self.otfnal.grids, key, None))
        # Make sure verbose and stdout don't accidentally change 
        # (i.e., in scanner mode)
        self.otfnal.verbose = self.verbose
        self.otfnal.stdout = self.stdout
        
    def kernel (self, mo=None, ci=None, **kwargs):
        self.otfnal.reset (mol=self.mol) # scanner mode safety
        self.e_mcscf, self.e_cas, self.ci, self.mo_coeff, self.mo_energy = \
            super().kernel (mo, ci, **kwargs)
        if isinstance (self, StateAverageMCSCFSolver):
            epdft = [self.energy_tot (root=ix) for ix in range (len (
                self.e_states))]
            self.e_mcscf = self.e_states
            self.fcisolver.e_states = [e_tot for e_tot, e_ot in epdft]
            self.e_ot = [e_ot for e_tot, e_ot in epdft]
            self.e_tot = np.dot (self.e_states, self.weights)
        else:
            self.e_tot, self.e_ot = self.energy_tot ()
        return (self.e_tot, self.e_ot, self.e_mcscf, self.e_cas, self.ci,
            self.mo_coeff, self.mo_energy)

    def dump_flags (self, verbose=None):
        super().dump_flags (verbose=verbose)
        log = logger.new_logger(self, verbose)
        log.info ('on-top pair density exchange-correlation functional: %s',
            self.otfnal.otxc)

    def get_pdft_veff (self, mo=None, ci=None, state=-1, casdm1s=None,
            casdm2=None, incl_coul=False, paaa_only=False, aaaa_only=False):
        ''' Get the 1- and 2-body MC-PDFT effective potentials for a set
            of mos and ci vectors

            Kwargs:
                mo : ndarray of shape (nao,nmo)
                    A full set of molecular orbital coefficients. Taken
                    from self if not provided
                ci : list or ndarray
                    CI vectors. Taken from self if not provided
                state : integer
                    Indexes a specific state in state-averaged
                    calculations. If negative, it generates a
                    state-averaged effective potential.
                casdm1s : ndarray of shape (2,ncas,ncas)
                    Spin-separated 1-RDM in the active space. Overrides
                    CI if and only if both this and casdm2 are provided
                casdm2 : ndarray of shape (ncas,ncas,ncas,ncas)
                    2-RDM in the active space. Overrides CI if and only
                    if both this and casdm1s are provided 
                incl_coul : logical
                    If true, includes the Coulomb repulsion energy in
                    the 1-body effective potential.
                paaa_only : logical
                    If true, only the paaa 2-body effective potential
                    elements are evaluated; the rest of ppaa are filled
                    with zeros.
                aaaa_only : logical
                    If true, only the aaaa 2-body effective potential
                    elements are evaluated; the rest of ppaa are filled
                    with zeros.

            Returns:
                veff1 : ndarray of shape (nao, nao)
                    1-body effective potential in the AO basis
                    May include classical Coulomb potential term (see
                    incl_coul kwarg)
                veff2 : pyscf.mcscf.mc_ao2mo._ERIS instance
                    Relevant 2-body effective potential in the MO basis
        ''' 
        t0 = (logger.process_clock (), logger.perf_counter ())
        if mo is None: mo = self.mo_coeff
        if ci is None: ci = self.ci
        ncore, ncas, nelecas = self.ncore, self.ncas, self.nelecas
        nocc = ncore + ncas

        if (casdm1s is not None) and (casdm2 is not None):
            mo_core = mo[:,:ncore]
            mo_cas = mo[:,ncore:nocc]
            dm1s = np.dot (mo_cas, casdm1s).transpose (1,0,2)
            dm1s = np.dot (dm1s, mo_cas.conj ().T)
            dm1s += (mo_core @ mo_core.conj ().T)[None,:,:]
            adm1s = casdm1s
            adm2 = get_2CDM_from_2RDM (casdm2, casdm1s)
        else:
            dm_list = self.make_rdms_mcpdft (mo_coeff=mo, ci=ci, state=state)
            dm1s, (adm1s, (adm2, _ss, _os)) = dm_list

        mo_cas = mo[:,ncore:][:,:ncas]
        pdft_veff1, pdft_veff2 = pdft_veff.kernel (self.otfnal, adm1s, 
            adm2, mo, ncore, ncas, max_memory=self.max_memory, 
            paaa_only=paaa_only, aaaa_only=aaaa_only)
        if self.verbose > logger.DEBUG:
            logger.debug (self, 'Warning: memory-intensive lazy kernel for '
                'pdft_veff initiated for testing purposes; reduce verbosity to'
                ' decrease memory footprint')
            # TODO: insufficient memory escape!
            pdft_veff1_test, _pdft_veff2_test = pdft_veff.lazy_kernel (
                self.otfnal, dm1s, adm2, mo_cas)
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
            pdft_veff1 += self._scf.get_j (self.mol, dm1s[0] + dm1s[1])
        logger.timer (self, 'get_pdft_veff', *t0)
        return pdft_veff1, pdft_veff2

    def nuc_grad_method (self):
        from mrh.my_pyscf.grad.mcpdft import Gradients
        return Gradients (self)

    def dip_moment (self, unit='Debye', state=None):
        from mrh.my_pyscf.prop.dip_moment.mcpdft import ElectricDipole
        is_sa = isinstance (self, StateAverageMCSCFSolver)
        if state is None and not is_sa:
            state = 0    
        if is_sa:
            # TODO: SA dipole moment unittests
            logger.warn (self, "State-averaged dipole moments are UNTESTED!")
        dip_obj =  ElectricDipole(self) 
        mol_dipole = dip_obj.kernel (state=state)
        return mol_dipole

    def get_energy_decomposition (self, mo_coeff=None, ci=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if ci is None: ci = self.ci
        return get_energy_decomposition (self, self.otfnal, mo_coeff=mo_coeff,
            ci=ci)

    def state_average (self, weights=(0.5,0.5)):
        # This is clumsy and hacky and should be fixed in pyscf.mcscf.addons
        # eventually rather than here
        return sapdft_grad_monkeypatch_(super ().state_average (
            weights=weights))

    def state_average_(self, weights=(0.5,0.5)):
        # This is clumsy and hacky and should be fixed in pyscf.mcscf.addons
        # eventually rather than here
        sapdft_grad_monkeypatch_(super ().state_average_(weights=weights))
        return self

    def state_average_mix (self, fcisolvers=None, weights=(0.5,0.5)):
        # This is clumsy and hacky and should be fixed in pyscf.mcscf.addons
        # eventually rather than here
        return sapdft_grad_monkeypatch_(state_average_mix (self, fcisolvers,
            weights))

    def state_average_mix_(self, fcisolvers=None, weights=(0.5,0.5)):
        # This is clumsy and hacky and should be fixed in pyscf.mcscf.addons
        # eventually rather than here
        sapdft_grad_monkeypatch_(state_average_mix_(self, fcisolvers, weights))
        return self

    def state_interaction (self, weights=(0.5,0.5), obj='CMS'):
        from mrh.my_pyscf.mcpdft.sipdft import state_interaction
        return state_interaction (self, weights=weights, obj=obj)

    @property
    def otxc (self):
        return self.otfnal.otxc

    @otxc.setter
    def otxc (self, x):
        self._init_ot_grids (x)

    make_rdms_mcpdft = make_rdms_mcpdft
    energy_mcwfn = energy_mcwfn
    energy_dft = energy_dft
    def energy_tot (self, ot=None, ci=None, root=-1, verbose=None):
        if ot is None: ot = self.otfnal
        if ci is None: ci = self.ci
        e_tot, e_ot = energy_tot (self, ot=ot, ci=ci, root=root,
            verbose=verbose)
        logger.note (self, 'MC-PDFT E = %s, Eot(%s) = %s', e_tot, ot.otxc,
            e_ot)
        return e_tot, e_ot

def get_mcpdft_child_class (mc, ot, ci_min='ecas', **kwargs):

    # Inheritance magic
    class PDFT (_PDFT, mc.__class__):
        if ci_min.lower () == 'epdft':
            if isinstance (mc, mc1step.CASSCF):
                casci=ci_scf.mc1step_casci # CASSCF CI step
                update_casdm=ci_scf.mc1step_update_casdm # innercycle CI update
            else:
                kernel=ci_scf.casci_kernel # CASCI
                _finalize=ci_scf.casci_finalize # I/O clarity

    pdft = PDFT (mc._scf, mc.ncas, mc.nelecas, my_ot=ot, **kwargs)
    _keys = pdft._keys.copy ()
    pdft.__dict__.update (mc.__dict__)
    pdft._keys = pdft._keys.union (_keys)
    return pdft

