import numpy as np
from scipy import linalg
from pyscf.lib import logger
from pyscf import mcpdft, lib
from pyscf.mcpdft import _dms
from mrh.my_pyscf.lassi.lassi import las_symm_tuple, iterate_subspace_blocks
from mrh.my_pyscf.lassi import op_o1

'''
This file is taken from pyscf-forge and adopted for the LAS wavefunctions.
'''


def weighted_average_densities(mc):
    """
	Compute the weighted average 1- and 2-electron LAS densities
	in the selected modal space
	"""
    casdm1s = [mc.make_one_casdm1s(mc.ci, state=state) for state in mc.statlis]
    casdm2 = [mc.make_one_casdm2(mc.ci, state=state) for state in mc.statlis]
    weights = [1 / len(mc.statlis), ] * len(mc.statlis)
    return (np.tensordot(weights, casdm1s, axes=1)), (np.tensordot(weights, casdm2, axes=1))


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
    e_veff1_j = np.tensordot(veff1 + hyb * 0.5 * vj, dm1)

    # Deal with 2-electron on-top potential energy
    e_veff2 = veff2.energy_core
    e_veff2 += np.tensordot(veff2.vhf_c[ncore:nocc, ncore:nocc], casdm1_0)
    e_veff2 += 0.5 * np.tensordot(veff2.papa[ncore:nocc, :, ncore:nocc, :], casdm2_0, axes=4)

    # h_nuc + E_ot - 1/2 g_pqrs D_pq D_rs - V_pq D_pq - 1/2 v_pqrs d_pqrs
    energy_core = hyb * mc.energy_nuc() + E_ot - e_veff1_j - e_veff2
    return energy_core


def transformed_h1e_for_cas(mc, E_ot, casdm1s_0, casdm2_0, hyb=1.0,
                            mo_coeff=None, ncas=None, ncore=None):
    """Compute the LAS one-particle L-PDFT Hamiltonian

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
    """
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
    casdm1s_0, casdm2_0 = mc.get_casdm12_0()

    mc.veff1, mc.veff2, E_ot = mc.get_pdft_veff(mo=mo_coeff, casdm1s=casdm1s_0,
                                                casdm2=casdm2_0, drop_mcwfn=True, incl_energy=True)

    # This is all standard procedure for generating the hamiltonian in PySCF
    h1, h0 = mc.get_h1lpdft(E_ot, casdm1s_0, casdm2_0, hyb=1.0 - cas_hyb)
    h2 = mc.get_h2lpdft()

    statesym, s2_states = las_symm_tuple(mc, verbose=0)

    # Initialize matrices
    e_roots = []
    s2_roots = []
    rootsym = []
    si = []
    s2_mat = []
    idx_allprods = []
    # Loop over symmetry blocks
    qn_lbls = ['neleca', 'nelecb', 'irrep']
    for it, (las1, sym, indices, indexed) in enumerate(iterate_subspace_blocks(mc, ci, statesym)):
        idx_space, idx_prod = indices
        ci_blk, nelec_blk = indexed
        idx_allprods.extend(list(np.where(idx_prod)[0]))
        lib.logger.info(mc, 'Build + diag H matrix L-PDFT-LASSI symmetry block %d\n'
                        + '{} = {}\n'.format(qn_lbls, sym)
                        + '(%d rootspaces; %d states)', it,
                        np.count_nonzero(idx_space),
                        np.count_nonzero(idx_prod))

        ham_blk, s2_blk, ovlp_blk = op_o1.ham(mc, h1, h2, ci_blk, nelec_blk)
        diag_idx = np.diag_indices_from(ham_blk)
        ham_blk[diag_idx] += h0 + cas_hyb * mc.e_roots

        try:
            e, c = linalg.eigh(ham_blk, b=ovlp_blk)
        except linalg.LinAlgError as err:
            ovlp_det = linalg.det(ovlp_blk)
            lc = 'checking if L-PDFT-LASSI basis has lindeps: |ovlp| = {:.6e}'.format(ovlp_det)
            lib.logger.info(las, 'Caught error %s, %s', str(err), lc)
            if ovlp_det < LINDEP_THRESH:
                x = canonical_orth_(ovlp_blk, thr=LINDEP_THRESH)
                lib.logger.info(las, '%d/%d linearly independent model states',
                                x.shape[1], x.shape[0])
                xhx = x.conj().T @ ham_blk @ x
                e, c = linalg.eigh(xhx)
                c = x @ c
            else:
                raise (err) from None

        s2_mat.append(s2_blk)
        si.append(c)
        s2_blk = c.conj().T @ s2_blk @ c
        lib.logger.debug2(mc, 'Block S**2 in adiabat basis:')
        lib.logger.debug2(mc, '{}'.format(s2_blk))
        e_roots.extend(list(e))
        s2_roots.extend(list(np.diag(s2_blk)))
        rootsym.extend([sym, ] * c.shape[1])

    idx_allprods = np.argsort(idx_allprods)
    si = linalg.block_diag(*si)[idx_allprods, :]
    s2_mat = linalg.block_diag(*s2_mat)[np.ix_(idx_allprods, idx_allprods)]
    idx = np.argsort(e_roots)
    rootsym = np.asarray(rootsym)[idx]
    s2_roots = np.asarray(s2_roots)[idx]
    si = si[:, idx]
    return ham_blk, e, si, rootsym, s2_roots, s2_mat


def kernel(mc, mo_coeff=None, ot=None, **kwargs):
    if ot is None: ot = mc.otfnal
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    mc.optimize_mcscf_(mo_coeff=mo_coeff, **kwargs)
    mc.lpdft_ham, mc.e_states, mc.si_pdft, mc.rootsym, mc.s2_roots, s2_mat = mc.make_lpdft_ham_(ot=ot)
    logger.debug(mc, f"L-PDFT Hamiltonian in LASSI Basis:\n{mc.get_lpdft_ham()}")

    logger.debug(mc, f"L-PDFT SI:\n{mc.si_pdft}")

    mc.e_tot = mc.e_states[0]
    logger.info(mc, 'LASSI-LPDFT eigenvalues (%d total):', len(mc.e_states))

    fmt_str = '{:2s}   {:>16s}  {:2s}  '
    col_lbls = ['ix', 'Energy', '<S**2>']
    logger.info(mc, fmt_str.format(*col_lbls))
    fmt_str = '{:2d}  {:16.10f}  {:6.3f}  '
    for ix, (er, s2r) in enumerate(zip(mc.e_states, mc.s2_roots)):
        row = [ix, er, s2r]
        logger.info(mc, fmt_str.format(*row))
        if ix >= 99 and mc.verbose < lib.logger.DEBUG:
            lib.logger.info(mc, ('Remaining %d eigenvalues truncated; '
                                 'increase verbosity to print them all'), len(mc.e_states) - 100)
            break

    return mc.e_tot, mc.e_states, mc.si_pdft, mc.s2_roots


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

        kernel(self, mo_coeff, ot=ot, verbose=log)

        return (
            self.e_tot, self.e_mcscf, self.e_cas, self.ci,
            self.mo_coeff, self.mo_energy)

    def get_lpdft_hcore_only(self, casdm1s_0, hyb=1.0):
        '''
        Returns the lpdft hcore AO integrals weighted by the
        hybridization factor. Excludes the MC-SCF (wfn) component.
        '''

        dm1s = _dms.casdm1s_to_dm1s(self, casdm1s=casdm1s_0)
        dm1 = dm1s[0] + dm1s[1]
        v_j = self._scf.get_j(dm=dm1)
        return hyb * self.get_hcore() + self.veff1 + hyb * v_j

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


def linear_multi_state(mc, **kwargs):
    ''' Build linearized multi-state MC-PDFT method object

    Args:
        mc : instance of class _PDFT

    Kwargs:
        weights : sequence of floats

    Returns:
        si : instance of class _LPDFT
    '''

    base_name = mc.__class__.__name__
    mcbase_class = mc.__class__

    class LPDFT(_LPDFT, mcbase_class):
        pass

    LPDFT.__name__ = "LIN" + base_name
    return LPDFT(mc)
