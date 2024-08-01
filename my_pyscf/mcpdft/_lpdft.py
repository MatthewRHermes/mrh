import numpy as np
from scipy import linalg
from pyscf.lib import logger
from pyscf.fci import direct_spin1
from pyscf import mcpdft
from pyscf.mcpdft import _dms

'''
This file is taken from pyscf-forge and adopted for the
LAS wavefunctions.
Author: Bhavnesh Jangid <jangidbhavnesh@uchicago.com>
'''


def weighted_average_densities(mc):
    '''
	Compute the weighted average 1- and 2-electron LAS densities
	in the selected modal space
	'''
    casdm1s = [mc.make_one_casdm1s(mc.ci, state=state) for state in mc.statlis]
    casdm2  = [mc.make_one_casdm2(mc.ci, state=state) for state in mc.statlis]
    weights = [1/len(mc.statlis),]*mc.statlis
    return (np.tensordot(weights, casdm1s, axes=1)),(np.tensordot(weights, casdm2, axes=1))


def get_lpdft_hconst(mc, E_ot, casdm1s_0, casdm2_0, hyb=1.0, ncas=None, ncore=None, veff1=None, veff2=None,
                     mo_coeff=None):
    ''' Compute h_const for the L-PDFT Hamiltonian

    Args:
        mc : instance of class _PDFT

        E_ot : float
            On-top energy

        casdm1s_0 : ndarray of shape (2, ncas, ncas)
            Spin-separated 1-RDM in the active space generated from expansion
            density.

        casdm2_0 : ndarray of shape (ncas, ncas, ncas, ncas)
            Spin-summed 2-RDM in the active space generated from expansion
            density.

    Kwargs:
        hyb : float
            Hybridization constant (lambda term)

        ncas : float
            Number of active space MOs

        ncore: float
            Number of core MOs

        veff1 : ndarray of shape (nao, nao)
            1-body effective potential in the AO basis computed using the
            zeroth-order densities.

        veff2 : pyscf.mcscf.mc_ao2mo._ERIS instance
            Relevant 2-body effective potential in the MO basis.

    Returns:
        Constant term h_const for the expansion term.
    '''
    if ncas is None: ncas = mc.ncas
    if ncore is None: ncore = mc.ncore
    if veff1 is None: veff1 = mc.veff1
    if veff2 is None: veff2 = mc.veff2
    if mo_coeff is None: mo_coeff = mc.mo_coeff

    nocc = ncore + ncas

    # Get the 1-RDM matrices
    casdm1_0 = casdm1s_0[0] + casdm1s_0[1]
    dm1s = _dms.casdm1s_to_dm1s(mc, casdm1s=casdm1s_0, mo_coeff=mo_coeff)
    dm1 = dm1s[0] + dm1s[1]

    # Coulomb interaction
    vj = mc._scf.get_j(dm=dm1)
    e_veff1_j = np.tensordot(veff1 + hyb*0.5*vj, dm1)

    # Deal with 2-electron on-top potential energy
    e_veff2 = veff2.energy_core
    e_veff2 += np.tensordot(veff2.vhf_c[ncore:nocc, ncore:nocc], casdm1_0)
    e_veff2 += 0.5 * np.tensordot(veff2.papa[ncore:nocc, :, ncore:nocc, :], casdm2_0, axes=4)

    # h_nuc + E_ot - 1/2 g_pqrs D_pq D_rs - V_pq D_pq - 1/2 v_pqrs d_pqrs
    energy_core = hyb * mc.energy_nuc() + E_ot - e_veff1_j - e_veff2
    return energy_core


def transformed_h1e_for_cas(mc, E_ot, casdm1s_0, casdm2_0, hyb=1.0,
                            mo_coeff=None, ncas=None, ncore=None):
    '''Compute the LAS one-particle L-PDFT Hamiltonian

    Args:
        mc : instance of a _PDFT object

        E_ot : float
            On-top energy

        casdm1s_0 : ndarray of shape (2,ncas,ncas)
            Spin-separated 1-RDM in the active space generated from expansion
            density

        casdm2_0 : ndarray of shape (ncas,ncas,ncas,ncas)
            Spin-summed 2-RDM in the active space generated from expansion
            density

        hyb : float
            Hybridization constant (lambda term)

        mo_coeff : ndarray of shape (nao,nmo)
            A full set of molecular orbital coefficients. Taken from self if
            not provided.

        ncas : int
            Number of active space molecular orbitals

        ncore : int
            Number of core molecular orbitals

    Returns:
        A tuple, the first is the effective one-electron linear PDFT
        Hamiltonian defined in CAS space, the second is the modified core
        energy.
    '''
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ncas is None: ncas = mc.ncas
    if ncore is None: ncore = mc.ncore

    nocc = ncore + ncas
    mo_core = mo_coeff[:, :ncore]
    mo_cas = mo_coeff[:, ncore:nocc]

    # h_pq + V_pq + J_pq all in AO integrals
    hcore_eff = mc.get_lpdft_hcore_only(casdm1s_0, hyb=hyb)
    energy_core = mc.get_lpdft_hconst(E_ot, casdm1s_0, casdm2_0, hyb)

    if mo_core.size != 0:
        core_dm = np.dot(mo_core, mo_core.conj().T) * 2
        # This is precomputed in MRH's ERIS object
        energy_core += mc.veff2.energy_core
        energy_core += np.tensordot(core_dm, hcore_eff).real

    h1eff = mo_cas.conj().T @ hcore_eff @ mo_cas
    # Add in the 2-electron portion that acts as a 1-electron operator
    h1eff += mc.veff2.vhf_c[ncore:nocc, ncore:nocc]

    return h1eff, energy_core


def get_transformed_h2eff_for_cas(mc, ncore=None, ncas=None):
    '''Compute the LAS two-particle linear PDFT Hamiltonian

    Args:
        ncore : int
            Number of core MOs

        ncas : int
            Number of active space MOs

    Returns:
        ndarray of shape (ncas,ncas,ncas,ncas) which contain v_vwxy
    '''
    if ncore is None: ncore = mc.ncore
    if ncas is None: ncas = mc.ncas
    nocc = ncore + ncas
    return mc.veff2.papa[ncore:nocc, :, ncore:nocc, :]


def make_lpdft_ham_(mc, mo_coeff=None, ci=None, ot=None):
    '''Compute the L-PDFT Hamiltonian

    Args:
        mo_coeff : ndarray of shape (nao, nmo)
            A full set of molecular orbital coefficients. Taken from self if
            not provided.

        ci : list of ndarrays of length nroots
            CI vectors should be from a converged CASSCF/CASCI calculation

        ot : an instance of on-top functional class - see otfnal.py

    Returns:
        lpdft_ham : ndarray of shape (nroots, nroots) or (nirreps, nroots, nroots)
            Linear approximation to the MC-PDFT energy expressed as a
            hamiltonian in the basis provided by the CI vectors. If
            StateAverageMix, then returns the block diagonal of the lpdft
            hamiltonian for each irrep.
    '''

    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci
    if ot is None: ot = mc.otfnal

    ot.reset(mol=mc.mol)

    spin = abs(mc.nelecas[0] - mc.nelecas[1])
    omega, _, hyb = ot._numint.rsh_and_hybrid_coeff(ot.otxc, spin=spin)
    if abs(omega) > 1e-11:
        raise NotImplementedError("range-separated on-top functionals")
    if abs(hyb[0] - hyb[1]) > 1e-11:
        raise NotImplementedError(
            "hybrid functionals with different exchange, correlations components")

    cas_hyb = hyb[0]

    ncas = mc.ncas
    casdm1s_0, casdm2_0 = mc.get_casdm12_0(ci=ci)

    mc.veff1, mc.veff2, E_ot = mc.get_pdft_veff(mo=mo_coeff, casdm1s=casdm1s_0,
                                        casdm2=casdm2_0, drop_mcwfn=True, incl_energy=True)

    # This is all standard procedure for generating the hamiltonian in PySCF
    h1, h0 = mc.get_h1lpdft(E_ot, casdm1s_0, casdm2_0, hyb=1.0-cas_hyb)
    h2 = mc.get_h2lpdft()
    
    # Using the h0, h1 and h2, I have to pass this to _eig_block of the lassi
    # 
    e, c, s2_blk = _eig_block (las1, e0, h1, h2, ci_blk, nelec_blk, sym, soc,
                                   orbsym, wfnsym, o0_memcheck, opt)
    # h2eff = direct_spin1.absorb_h1e(h1, h2, ncas, mc.nelecas, 0.5)

    # def construct_ham_slice(solver, slice, nelecas):
        # ci_irrep = ci[slice]
        # if hasattr(solver, "orbsym"):
            # solver.orbsym = mc.fcisolver.orbsym

        # hc_all_irrep = [solver.contract_2e(h2eff, c, ncas, nelecas) for c in ci_irrep]
        # lpdft_irrep = np.tensordot(ci_irrep, hc_all_irrep, axes=((1, 2), (1, 2)))
        # diag_idx = np.diag_indices_from(lpdft_irrep)
        # lpdft_irrep[diag_idx] += h0 + cas_hyb * mc.e_mcscf[slice]
        # return lpdft_irrep

    # if not isinstance(mc, _LPDFTMix):
        # return construct_ham_slice(direct_spin1, slice(0, len(ci)), mc.nelecas)

    # # We have a StateAverageMix Solver
    # mc._irrep_slices = []
    # start = 0
    # for solver in mc.fcisolver.fcisolvers:
        # end = start + solver.nroots
        # mc._irrep_slices.append(slice(start, end))
        # start = end

    # return [construct_ham_slice(s, irrep, mc.fcisolver._get_nelec(s, mc.nelecas))
            # for s, irrep in zip(mc.fcisolver.fcisolvers, mc._irrep_slices)]


def kernel(mc, mo_coeff=None, ci0=None, ot=None, **kwargs):
    if ot is None: ot = mc.otfnal
    if mo_coeff is None: mo_coeff = mc.mo_coeff

    mc.optimize_mcscf_(mo_coeff=mo_coeff, ci0=ci0)
    mc.ci_mcscf = mc.ci
    mc.lpdft_ham = mc.make_lpdft_ham_(ot=ot)
    logger.debug(mc, f"L-PDFT Hamiltonian in MC-SCF Basis:\n{mc.get_lpdft_ham()}")

    if hasattr(mc, "_irrep_slices"):
        e_states, si_pdft = zip(*map(mc._eig_si, mc.lpdft_ham))
        mc.e_states = np.concatenate(e_states)
        mc.si_pdft = linalg.block_diag(*si_pdft)

    else:
        mc.e_states, mc.si_pdft = mc._eig_si(mc.lpdft_ham)

    logger.debug(mc, f"L-PDFT SI:\n{mc.si_pdft}")

    mc.e_tot = np.dot(mc.e_states, mc.weights)
    mc.ci = mc._get_ci_adiabats()

    return (
        mc.e_tot, mc.e_mcscf, mc.e_cas, mc.ci,
        mc.mo_coeff, mc.mo_energy)


class _LPDFT(mcpdft.MultiStateMCPDFTSolver):
    '''Linerized PDFT

    Saved Results

        e_tot : float
            Weighted-average L-PDFT final energy
        e_states : ndarray of shape (nroots)
            L-PDFT final energies of the adiabatic states
        ci : list of length (nroots) of ndarrays
            CI vectors in the optimized adiabatic basis of MC-SCF. Related to
            the L-PDFT adiabat CI vectors by the expansion coefficients
            ``si_pdft''.
        si_pdft : ndarray of shape (nroots, nroots)
            Expansion coefficients of the L-PDFT adiabats in terms of the
            optimized
            MC-SCF adiabats
        e_mcscf : ndarray of shape (nroots)
            Energies of the MC-SCF adiabatic states
        lpdft_ham : ndarray of shape (nroots, nroots)
            L-PDFT Hamiltonian in the MC-SCF adiabatic basis
        veff1 : ndarray of shape (nao, nao)
            1-body effective potential in the AO basis computed using the
            zeroth-order densities.
        veff2 : pyscf.mcscf.mc_ao2mo._ERIS instance
            Relevant 2-body effective potential in the MO basis.
    '''

    def __init__(self, mc):
        self.__dict__.update(mc.__dict__)
        keys = set(('lpdft_ham', 'si_pdft', 'veff1', 'veff2'))
        self.lpdft_ham = None
        self.si_pdft = None
        self.veff1 = None
        self.veff2 = None
        self._e_states = None
        self._keys = set(self.__dict__.keys()).union(keys)

    @property
    def e_states(self):
        if self._in_mcscf_env:
            return self.fcisolver.e_states

        else:
            return self._e_states

    @e_states.setter
    def e_states(self, x):
        self._e_states = x

    make_lpdft_ham_ = make_lpdft_ham_
    make_lpdft_ham_.__doc__ = make_lpdft_ham_.__doc__

    get_lpdft_hconst = get_lpdft_hconst
    get_lpdft_hconst.__doc__ = get_lpdft_hconst.__doc__

    get_h1lpdft = transformed_h1e_for_cas
    get_h1lpdft.__doc__ = transformed_h1e_for_cas.__doc__

    get_h2lpdft = get_transformed_h2eff_for_cas
    get_h2lpdft.__doc__ = get_transformed_h2eff_for_cas.__doc__

    get_casdm12_0 = weighted_average_densities
    get_casdm12_0.__doc__ = weighted_average_densities.__doc__

    def get_lpdft_diag(self):
        '''Diagonal elements of the L-PDFT Hamiltonian matrix
            (H_00^L-PDFT, H_11^L-PDFT, H_22^L-PDFT, ...)

        Returns:
            lpdft_diag : ndarray of shape (nroots)
                Contains the linear approximation to the MC-PDFT energy. These
                are also the diagonal elements of the L-PDFT Hamiltonian
                matrix.
        '''
        return np.diagonal(self.lpdft_ham).copy()

    def get_lpdft_ham(self):
        '''The L-PDFT effective Hamiltonian matrix

            Returns:
                lpdft_ham : ndarray of shape (nroots, nroots)
                    Contains L-PDFT Hamiltonian elements on the off-diagonals
                    and PDFT approx energies on the diagonals
                '''
        return self.lpdft_ham

    def kernel(self, mo_coeff=None, ci0=None, ot=None, verbose=None):
        '''
        Returns:
            6 elements, they are
            total energy,
            the MCSCF energies,
            the active space CI energy,
            the active space FCI wave function coefficients,
            the MCSCF canonical orbital coefficients,
            the MCSCF canonical orbital energies

        They are attributes of the QLPDFT object, which can be accessed by
        .e_tot, .e_mcscf, .e_cas, .ci, .mo_coeff, .mo_energy
        '''
        if ot is None: ot = self.otfnal
        ot.reset(mol=self.mol)  # scanner mode safety

        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        else:
            self.mo_coeff = mo_coeff

        log = logger.new_logger(self, verbose)

        if ci0 is None and isinstance(getattr(self, 'ci', None), list):
            ci0 = [c.copy() for c in self.ci]

        kernel(self, mo_coeff, ci0, ot=ot, verbose=log)
        self._finalize_lin()
        return (
            self.e_tot, self.e_mcscf, self.e_cas, self.ci,
            self.mo_coeff, self.mo_energy)

    def _finalize_lin(self):
        log = logger.Logger(self.stdout, self.verbose)
        nroots = len(self.e_states)
        log.note("%s (final) states:", self.__class__.__name__)
        if log.verbose >= logger.NOTE and getattr(self.fcisolver, 'spin_square',
                                                  None):
            ss = self.fcisolver.states_spin_square(self.ci, self.ncas, self.nelecas)[0]

            for i in range(nroots):
                log.note('  State %d weight %g  ELPDFT = %.15g  S^2 = %.7f',
                         i, self.weights[i], self.e_states[i], ss[i])

        else:
            for i in range(nroots):
                log.note('  State %d weight %g  ELPDFT = %.15g', i,
                         self.weights[i], self.e_states[i])

    def _get_ci_adiabats(self, ci_mcscf=None):
        '''Get the CI vertors in eigenbasis of L-PDFT Hamiltonian

            Kwargs:
                ci : list of length nroots
                    MC-SCF ci vectors; defaults to self.ci_mcscf

            Returns:
                ci : list of length nroots
                    CI vectors in basis of L-PDFT Hamiltonian eigenvectors
        '''
        if ci_mcscf is None: ci_mcscf = self.ci_mcscf
        return list(np.tensordot(self.si_pdft.T, np.asarray(ci_mcscf), axes=1))

    def _eig_si(self, ham):
        return linalg.eigh(ham)

    def get_lpdft_hcore_only(self, casdm1s_0, hyb=1.0):
        '''
        Returns the lpdft hcore AO integrals weighted by the
        hybridization factor. Excludes the MC-SCF (wfn) component.
        '''

        dm1s = _dms.casdm1s_to_dm1s(self, casdm1s=casdm1s_0)
        dm1 = dm1s[0] + dm1s[1]
        v_j = self._scf.get_j(dm=dm1)
        return hyb*self.get_hcore() + self.veff1 + hyb * v_j


    def get_lpdft_hcore(self, casdm1s_0=None):
        '''
        Returns the full lpdft hcore AO integrals. Includes the MC-SCF
        (wfn) component for hybrid functionals.
        '''
        if casdm1s_0 is None:
            casdm1s_0 = self.get_casdm12_0()[0]

        spin = abs(self.nelecas[0] - self.nelecas[1])
        cas_hyb = self.otfnal._numint.rsh_and_hybrid_coeff(self.otfnal.otxc, spin=spin)[2]
        hyb = 1.0 - cas_hyb[0]

        return cas_hyb[0] * self.get_hcore() + self.get_lpdft_hcore_only(casdm1s_0, hyb=hyb)

    def nuc_grad_method(self, state=None):
        from pyscf.mcscf import mc1step
        from pyscf.mcscf.df import _DFCASSCF
        if not isinstance(self, mc1step.CASSCF):
            raise NotImplementedError("CASCI-based LPDFT nuclear gradients")
        elif getattr(self, 'frozen', None) is not None:
            raise NotImplementedError("LPDFT nuclear gradients with frozen orbitals")
        elif isinstance(self, _DFCASSCF):
            from pyscf.df.grad.lpdft import Gradients
        else:
            from pyscf.grad.lpdft import Gradients

        return Gradients(self, state=state)


class _LPDFTMix(_LPDFT):
    '''State Averaged Mixed Linerized PDFT

    Saved Results

        e_tot : float
            Weighted-average L-PDFT final energy
        e_states : ndarray of shape (nroots)
            L-PDFT final energies of the adiabatic states
        ci : list of length (nroots) of ndarrays
            CI vectors in the optimized adiabatic basis of MC-SCF. Related to the
            L-PDFT adiabat CI vectors by the expansion coefficients ``si_pdft''.
        si_pdft : ndarray of shape (nroots, nroots)
            Expansion coefficients of the L-PDFT adiabats in terms of the optimized
            MC-SCF adiabats
        e_mcscf : ndarray of shape (nroots)
            Energies of the MC-SCF adiabatic states
        lpdft_ham : list of ndarray of shape (nirreps, nroots, nroots)
            L-PDFT Hamiltonian in the MC-SCF adiabatic basis within each irrep
    '''

    def __init__(self, mc):
        super().__init__(mc)
        # Holds the irrep slices for when we need to index into various quantities
        self._irrep_slices = None

    def get_lpdft_diag(self):
        '''Diagonal elements of the L-PDFT Hamiltonian matrix
            (H_00^L-PDFT, H_11^L-PDFT, H_22^L-PDFT, ...)

        Returns:
            lpdft_diag : ndarray of shape (nroots)
                Contains the linear approximation to the MC-PDFT energy. These
                are also the diagonal elements of the L-PDFT Hamiltonian
                matrix.
        '''
        return np.concatenate([np.diagonal(irrep_ham).copy() for irrep_ham in self.lpdft_ham])

    def get_lpdft_ham(self):
        '''The L-PDFT effective Hamiltonian matrix

            Returns:
                lpdft_ham : ndarray of shape (nroots, nroots)
                    Contains L-PDFT Hamiltonian elements on the off-diagonals
                    and PDFT approx energies on the diagonals
        '''
        return linalg.block_diag(*self.lpdft_ham)

    def _get_ci_adiabats(self, ci_mcscf=None):
        '''Get the CI vertors in eigenbasis of L-PDFT Hamiltonian

            Kwargs:
                ci : list of length nroots
                    MC-SCF ci vectors; defaults to self.ci_mcscf

            Returns:
                ci : list of length nroots
                    CI vectors in basis of L-PDFT Hamiltonian eigenvectors
        '''
        if ci_mcscf is None: ci_mcscf = self.ci_mcscf
        adiabat_ci = [np.tensordot(self.si_pdft[irrep_slice, irrep_slice],
                                   np.asarray(ci_mcscf[irrep_slice]), axes=1) for irrep_slice in self._irrep_slices]
        # Flattens it
        return [c for ci_irrep in adiabat_ci for c in ci_irrep]

    def nuc_grad_method(self, state=None):
        raise NotImplementedError("MultiState Mix LPDFT nuclear gradients")


def linear_multi_state(mc, weights=(0.5, 0.5), **kwargs):
    ''' Build linearized multi-state MC-PDFT method object

    Args:
        mc : instance of class _PDFT

    Kwargs:
        weights : sequence of floats

    Returns:
        si : instance of class _LPDFT
    '''
    from pyscf.mcscf.addons import StateAverageMCSCFSolver, \
        StateAverageMixFCISolver

    if isinstance(mc, mcpdft.MultiStateMCPDFTSolver):
        raise RuntimeError('already a multi-state PDFT solver')

    if isinstance(mc.fcisolver, StateAverageMixFCISolver):
        raise RuntimeError("state-average mix type")

    if not isinstance(mc, StateAverageMCSCFSolver):
        base_name = mc.__class__.__name__
        mc = mc.state_average(weights=weights, **kwargs)

    else:
        base_name = mc.__class__.bases__[0].__name__

    mcbase_class = mc.__class__

    class LPDFT(_LPDFT, mcbase_class):
        pass

    LPDFT.__name__ = "LIN" + base_name
    return LPDFT(mc)

def linear_multi_state_mix(mc, fcisolvers, weights=(0.5, 0.5), **kwargs):
    ''' Build SA Mix linearized multi-state MC-PDFT method object

    Args:
        mc : instance of class _PDFT

        fcisolvers : fcisolvers to construct StateAverageMixSolver with

    Kwargs:
        weights : sequence of floats

    Returns:
        si : instance of class _LPDFT
    '''
    from pyscf.mcscf.addons import StateAverageMCSCFSolver, \
        StateAverageMixFCISolver

    if isinstance(mc, mcpdft.MultiStateMCPDFTSolver):
        raise RuntimeError('already a multi-state PDFT solver')

    if not isinstance(mc, StateAverageMCSCFSolver):
        base_name = mc.__class__.__name__
        mc = mc.state_average_mix(fcisolvers, weights=weights, **kwargs)

    elif not isinstance(mc.fcisolver, StateAverageMixFCISolver):
        raise RuntimeError("already a StateAverageMCSCF solver")

    else:
        base_name = mc.__class__.bases__[0].__name__

    mcbase_class = mc.__class__

    class LPDFT(_LPDFTMix, mcbase_class):
        pass

    LPDFT.__name__ = "LIN" + base_name
    return LPDFT(mc)


'''
These two functions are required for the diagonalization of the 
LASSI States in the L-PDFT Hamiltonian format
'''

from mrh.my_pyscf.lassi.lassi import las_symm_tuple, iterate_subspace_blocks

def lassi (las, e0, h1, h2, mo_coeff=None, ci=None, veff_c=None, h2eff_sub=None, orbsym=None, soc=False,
           break_symmetry=False, opt=1):
    ''' Diagonalize the state-interaction matrix of LASSCF '''
    if mo_coeff is None: mo_coeff = las.mo_coeff
    if ci is None: ci = las.ci
    if orbsym is None: 
        orbsym = getattr (las.mo_coeff, 'orbsym', None)
        if orbsym is None and callable (getattr (las, 'label_symmetry_', None)):
            orbsym = las.label_symmetry_(las.mo_coeff).orbsym
        if orbsym is not None:
            orbsym = orbsym[las.ncore:las.ncore+las.ncas]
    o0_memcheck = op_o0.memcheck (las, ci, soc=soc)
    if opt == 0 and o0_memcheck == False:
        raise RuntimeError ('Insufficient memory to use o0 LASSI algorithm')

    # h0, h1 and h2 is from L-PDFT

    # Symmetry tuple: neleca, nelecb, irrep
    statesym, s2_states = las_symm_tuple (las, break_spin=soc, break_symmetry=break_symmetry)

    # Initialize matrices
    e_roots = []
    s2_roots = []
    rootsym = []
    si = []
    s2_mat = []
    idx_allprods = []
    dtype = complex if soc else np.float64

    # Loop over symmetry blocks
    qn_lbls = ['nelec',] if soc else ['neleca','nelecb',]
    if not break_symmetry: qn_lbls.append ('irrep')
    for it, (las1,sym,indices,indexed) in enumerate (iterate_subspace_blocks(las,ci,statesym)):
        idx_space, idx_prod = indices
        ci_blk, nelec_blk = indexed
        idx_allprods.extend (list(np.where(idx_prod)[0]))
        lib.logger.info (las, 'Build + diag H matrix LASSI symmetry block %d\n'
                         + '{} = {}\n'.format (qn_lbls, sym)
                         + '(%d rootspaces; %d states)', it,
                         np.count_nonzero (idx_space), 
                         np.count_nonzero (idx_prod))
        if np.count_nonzero (idx_prod) == 1:
            lib.logger.debug (las, 'Only one state in this symmetry block')
            e_roots.extend (las1.e_states - e0)
            si.append (np.ones ((1,1), dtype=dtype))
            s2_mat.append (s2_states[idx_space]*np.ones((1,1)))
            s2_roots.extend (s2_states[idx_space])
            rootsym.extend ([sym,])
            continue
        wfnsym = None if break_symmetry else sym[-1]
        e, c, s2_blk = _eig_block (las1, e0, h1, h2, ci_blk, nelec_blk, sym, soc,
                                   orbsym, wfnsym, o0_memcheck, opt)
        s2_mat.append (s2_blk)
        si.append (c)
        s2_blk = c.conj ().T @ s2_blk @ c
        lib.logger.debug2 (las, 'Block S**2 in adiabat basis:')
        lib.logger.debug2 (las, '{}'.format (s2_blk))
        e_roots.extend (list(e))
        s2_roots.extend (list (np.diag (s2_blk)))
        rootsym.extend ([sym,]*c.shape[1])

    # The matrix blocks were evaluated in idx_allprods order
    # Therefore, I need to ~invert~ idx_allprods to get the proper order
    idx_allprods = np.argsort (idx_allprods)
    si = linalg.block_diag (*si)[idx_allprods,:]
    s2_mat = linalg.block_diag (*s2_mat)[np.ix_(idx_allprods,idx_allprods)]

    # Sort results by energy
    idx = np.argsort (e_roots)
    rootsym = np.asarray (rootsym)[idx]
    e_roots = np.asarray (e_roots)[idx] + e0
    s2_roots = np.asarray (s2_roots)[idx]
    if soc == False:
        nelec_roots = [tuple(rs[0:2]) for rs in rootsym]
    else:
        nelec_roots = [rs[0] for rs in rootsym]
    if break_symmetry:
        wfnsym_roots = [None for rs in rootsym]
    else:
        wfnsym_roots = [rs[-1] for rs in rootsym]

    # Results tagged on si array....
    si = si[:,idx]
    si = tag_array (si, s2=s2_roots, s2_mat=s2_mat, nelec=nelec_roots, wfnsym=wfnsym_roots,
                    rootsym=rootsym, break_symmetry=break_symmetry, soc=soc)

    # I/O
    lib.logger.info (las, 'LASSI eigenvalues (%d total):', len (e_roots))
    fmt_str = ' {:2s}  {:>16s}  {:6s}  '
    col_lbls = ['Nelec'] if soc else ['Neleca','Nelecb']
    if not break_symmetry: col_lbls.append ('Wfnsym')
    fmt_str += '  '.join (['{:6s}',]*len(col_lbls))
    col_lbls = ['ix','Energy','<S**2>'] + col_lbls
    lib.logger.info (las, fmt_str.format (*col_lbls))
    fmt_str = ' {:2d}  {:16.10f}  {:6.3f}  '
    col_fmts = ['{:6d}',]*(2-int(soc))
    if not break_symmetry: col_fmts.append ('{:>6s}')
    fmt_str += '  '.join (col_fmts)
    for ix, (er, s2r, rsym) in enumerate (zip (e_roots, s2_roots, rootsym)):
        if np.iscomplexobj (s2r):
            assert (abs (s2r.imag) < 1e-8)
            s2r = s2r.real
        nelec = rsym[0:1] if soc else rsym[:2]
        row = [ix,er,s2r] + list (nelec)
        if not break_symmetry: row.append (symm.irrep_id2name (las.mol.groupname, rsym[-1]))
        lib.logger.info (las, fmt_str.format (*row))
        if ix>=99 and las.verbose < lib.logger.DEBUG:
            lib.logger.info (las, ('Remaining %d eigenvalues truncated; '
                                   'increase verbosity to print them all'), len (e_roots)-100)
            break
    return e_roots, si

def _eig_block (las, e0, h1, h2, ci_blk, nelec_blk, rootsym, soc, orbsym, wfnsym, o0_memcheck, opt):
    # TODO: simplify
    t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
    if (las.verbose > lib.logger.INFO) and (o0_memcheck):
        ham_ref, s2_ref, ovlp_ref = op_o0.ham (las, h1, h2, ci_blk, nelec_blk, soc=soc,
                                               orbsym=orbsym, wfnsym=wfnsym)
        t0 = lib.logger.timer (las, 'LASSI diagonalizer rootsym {} CI algorithm'.format (
            rootsym), *t0)

        h1_sf = h1
        if soc:
            h1_sf = (h1[0:las.ncas,0:las.ncas]
                     - h1[las.ncas:2*las.ncas,las.ncas:2*las.ncas]).real/2
        ham_blk, s2_blk, ovlp_blk = op_o1.ham (las, h1_sf, h2, ci_blk, nelec_blk, orbsym=orbsym,
                                               wfnsym=wfnsym)
        t0 = lib.logger.timer (las, 'LASSI diagonalizer rootsym {} TDM algorithm'.format (
            rootsym), *t0)
        lib.logger.debug (las,
            'LASSI diagonalizer rootsym {}: ham o0-o1 algorithm disagreement = {}'.format (
                rootsym, linalg.norm (ham_blk - ham_ref))) 
        lib.logger.debug (las,
            'LASSI diagonalizer rootsym {}: S2 o0-o1 algorithm disagreement = {}'.format (
                rootsym, linalg.norm (s2_blk - s2_ref))) 
        lib.logger.debug (las,
            'LASSI diagonalizer rootsym {}: ovlp o0-o1 algorithm disagreement = {}'.format (
                rootsym, linalg.norm (ovlp_blk - ovlp_ref))) 
        errvec = np.concatenate ([(ham_blk-ham_ref).ravel (), (s2_blk-s2_ref).ravel (),
                                  (ovlp_blk-ovlp_ref).ravel ()])
        if np.amax (np.abs (errvec)) > 1e-8 and soc == False: # tmp until SOC in op_o1
            raise LASSIOop01DisagreementError ("Hamiltonian + S2 + Ovlp", errvec)
        if opt == 0:
            ham_blk = ham_ref
            s2_blk = s2_ref
            ovlp_blk = ovlp_ref
    else:
        if (las.verbose > lib.logger.INFO): lib.logger.debug (
            las, 'Insufficient memory to test against o0 LASSI algorithm')
        ham_blk, s2_blk, ovlp_blk = op[opt].ham (las, h1, h2, ci_blk, nelec_blk, soc=soc,
                                                 orbsym=orbsym, wfnsym=wfnsym)
        t0 = lib.logger.timer (las, 'LASSI H build rootsym {}'.format (rootsym), *t0)
    log_debug = lib.logger.debug2 if las.nroots>10 else lib.logger.debug
    if np.iscomplexobj (ham_blk):
        log_debug (las, 'Block Hamiltonian - ecore (real):')
        log_debug (las, '{}'.format (ham_blk.real.round (8)))
        log_debug (las, 'Block Hamiltonian - ecore (imag):')
        log_debug (las, '{}'.format (ham_blk.imag.round (8)))
    else:
        log_debug (las, 'Block Hamiltonian - ecore:')
        log_debug (las, '{}'.format (ham_blk.round (8)))
    log_debug (las, 'Block S**2:')
    log_debug (las, '{}'.format (s2_blk.round (8)))
    log_debug (las, 'Block overlap matrix:')
    log_debug (las, '{}'.format (ovlp_blk.round (8)))
    # Error catch: diagonal Hamiltonian elements
    # This diagnostic is simply not valid for local excitations;
    # the energies aren't supposed to be additive
    lroots = get_lroots (ci_blk)
    e_states_meaningful = not getattr (las, 'e_states_meaningless', False)
    e_states_meaningful &= np.all (lroots==1)
    e_states_meaningful &= not (soc) # TODO: fix?
    if e_states_meaningful:
        diag_test = np.diag (ham_blk)
        diag_ref = las.e_states - e0
        maxerr = np.max (np.abs (diag_test-diag_ref))
        if maxerr>1e-5:
            lib.logger.debug (las, '{:>13s} {:>13s} {:>13s}'.format ('Diagonal', 'Reference',
                                                                     'Error'))
            for ix, (test, ref) in enumerate (zip (diag_test, diag_ref)):
                lib.logger.debug (las, '{:13.6e} {:13.6e} {:13.6e}'.format (test, ref, test-ref))
            lib.logger.warn (las, 'LAS states in basis may not be converged (%s = %e)',
                             'max(|Hdiag-e_states|)', maxerr)
    # Error catch: linear dependencies in basis
    try:
        e, c = linalg.eigh (ham_blk, b=ovlp_blk)
    except linalg.LinAlgError as err:
        ovlp_det = linalg.det (ovlp_blk)
        lc = 'checking if LASSI basis has lindeps: |ovlp| = {:.6e}'.format (ovlp_det)
        lib.logger.info (las, 'Caught error %s, %s', str (err), lc)
        if ovlp_det < LINDEP_THRESH:
            x = canonical_orth_(ovlp_blk, thr=LINDEP_THRESH)
            lib.logger.info (las, '%d/%d linearly independent model states',
                             x.shape[1], x.shape[0])
            xhx = x.conj ().T @ ham_blk @ x
            e, c = linalg.eigh (xhx)
            c = x @ c
        else: raise (err) from None
    return e, c, s2_blk
    
    
    
    

