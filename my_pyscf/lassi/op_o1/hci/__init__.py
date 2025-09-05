import numpy as np
from scipy import linalg
from pyscf import lib
from pyscf.lib import logger, param
from pyscf.fci import cistring 
from mrh.my_pyscf.lassi.op_o1 import stdm, frag, hams2ovlp, hsi, rdm
from mrh.my_pyscf.lassi.op_o1.utilities import *
from mrh.my_pyscf.lassi.citools import get_lroots, hci_dot_sivecs, hci_dot_sivecs_ij
from mrh.my_pyscf.lassi.op_o1.hci.chc import ContractHamCI_CHC
from mrh.my_pyscf.lassi.op_o1.hci.chc import gen_contract_ham_ci_const
from mrh.my_pyscf.lassi.op_o1.hci.schcs import ContractHamCI_SHS

def ContractHamCI (las, ints, nlas, lroots, h0, h1, h2, si_bra=None, si_ket=None,
                   mask_bra_space=None, mask_ket_space=None, pt_order=None, do_pt_order=None,
                   add_transpose=False, accum=None, log=None, max_memory=param.MAX_MEMORY,
                   dtype=np.float64):
    if si_bra is None and si_ket is None:
        return ContractHamCI_CHC (las, ints, nlas, lroots, h0, h1, h2,
                                  mask_bra_space=mask_bra_space,
                                  mask_ket_space=mask_ket_space,
                                  pt_order=pt_order, do_pt_order=do_pt_order,
                                  log=log, max_memory=max_memory, dtype=np.float64)
    elif (si_bra is None) or (si_ket is None):
        class ContractHamCI (ContractHamCI_CHC):
            def kernel (self):
                hci, t0 = super ().kernel ()
                hci = hci_dot_sivecs (hci, si_bra, si_ket, self.lroots)
                return hci, t0
        return ContractHamCI (las, ints, nlas, lroots, h0, h1, h2,
                              mask_bra_space=mask_bra_space,
                              mask_ket_space=mask_ket_space,
                              pt_order=pt_order, do_pt_order=do_pt_order,
                              log=log, max_memory=param.MAX_MEMORY, dtype=np.float64)
    else:
        return ContractHamCI_SHS (las, ints, nlas, lroots, h0, h1, h2, si_bra,
                                  si_ket, mask_bra_space=mask_bra_space,
                                  mask_ket_space=mask_ket_space,
                                  pt_order=pt_order, do_pt_order=do_pt_order,
                                  add_transpose=add_transpose, accum=accum, log=log,
                                  max_memory=param.MAX_MEMORY, dtype=np.float64)

def contract_ham_ci (las, h1, h2, ci_fr, nelec_frs, si_bra=None, si_ket=None, ci_fr_bra=None,
                     nelec_frs_bra=None, h0=0, soc=0, sum_bra=False, orbsym=None, wfnsym=None,
                     pt_order=None, do_pt_order=None, accum=None, add_transpose=False,
                     verbose=None):
    '''Evaluate the action of the state interaction Hamiltonian on a set of ket CI vectors,
    projected onto a basis of bra CI vectors, leaving one fragment of the bra uncontracted.

    Args:
        las : instance of class LASSCF
        h1 : ndarray of shape (ncas, ncas)
            Spin-orbit-free one-body CAS Hamiltonian
        h2 : ndarray of shape (ncas, ncas, ncas, ncas)
            Spin-orbit-free two-body CAS Hamiltonian
        ci_fr : nested list of shape (nfrags, nroots)
            Contains CI vectors; element [i,j] is ndarray of shape
            (ndeta_ket[i,j],ndetb_ket[i,j])
        nelec_frs : ndarray of shape (nfrags, nroots, 2)
            Number of electrons of each spin in each rootspace in each
            fragment

    Kwargs:
        si_bra : ndarray of shape (ndim_bra, nroots_si)
            SI vectors for the bra. If provided, the p dimension on the return object is contracted
        si_ket : ndarray of shape (ndim_ket, nroots_si)
            SI vectors for the bra. If provided, the q dimension on the return object is contracted
        ci_fr_bra : nested list of shape (nfrags, nroots_bra)
            Contains CI vectors for the bra; element [i,j] is ndarray of shape
            (ndeta_bra[i,j],ndetb_bra[i,j]). Defaults to ci_fr.
        nelec_frs_bra : ndarray of shape (nfrags, nroots_bra, 2)
            Number of electrons of each spin in each
            fragment for the bra vectors. Defaults to nelec_frs.
        soc : integer
            Order of spin-orbit coupling included in the Hamiltonian
        h0 : float
            Constant term in the Hamiltonian
        sum_bra : logical
            If true and both si_bra and si_ket are provided, then equivalent bra rootspaces are
            summed together and only one of the equivalent set of vectors is nonzero. Otherwise,
            if si_bra and si_ket are both provided, all bra rootspaces are forcibly considered
            distinct. This is necessary for a certain unit test but it hurts performance.
        orbsym : list of int of length (ncas)
            Irrep ID for each orbital
        wfnsym : int
            Irrep ID for target matrix block

    Returns:
        hket_fr : nested list of shape (nfrags, nroots_bra)
            If si_bra and si_ket are both provided, then element i,j is an ndarray of shape
            [nroots_si,] + list (ci_fr_bra[i][j].shape).
            Otherwise, element i,j is an ndarray of shape (ndim_bra//ci_fr_bra[i][j].shape[0],
            ndeta_bra[i,j],ndetb_bra[i,j],ndim_ket).
    '''
    if verbose is None: verbose = las.verbose
    log = lib.logger.new_logger (las, verbose)
    nlas = las.ncas_sub
    ci_fr_ket = ci_fr
    nelec_frs_ket = nelec_frs
    nfrags, nket = nelec_frs.shape[:2]
    if ci_fr_bra is None and nelec_frs_bra is None:
        ci = ci_fr
        nroots = nbra = nket
        mask_bra_space = mask_ket_space = list (range (nroots))
        mask_ints = None
    else:
        nbra = nelec_frs_bra.shape[1]
        ci = [ci_r_ket + ci_r_bra for ci_r_bra, ci_r_ket in zip (ci_fr_bra, ci_fr_ket)]
        nelec_frs = np.append (nelec_frs_ket, nelec_frs_bra, axis=1)
        nroots = nbra + nket
        mask_bra_space = list (range (nket,nroots))
        mask_ket_space = list (range (nket))
        mask_ints = np.zeros ((nroots,nroots), dtype=bool)
        mask_ints[np.ix_(mask_bra_space,mask_ket_space)] = True
    discriminator = np.zeros (nroots, dtype=int)
    si_bra_is1d = si_ket_is1d = False
    if si_bra is not None:
        si_bra_is1d = si_bra.ndim==1
        if si_bra_is1d: si_bra = si_bra[:,None]
    if si_ket is not None:
        si_ket_is1d = si_ket.ndim==1
        if si_ket_is1d: si_ket = si_ket[:,None]
    if si_bra is not None and si_ket is not None:
        assert (si_bra.shape[1] == si_ket.shape[1])
        si_ket_is1d = False
        discriminator[mask_bra_space] = 1
        if not sum_bra:
            discriminator[mask_bra_space] += np.arange (nbra, dtype=int)

    # First pass: single-fragment intermediates
    ints, lroots = frag.make_ints (las, ci, nelec_frs, nlas=nlas,
                                                  screen_linequiv=False,
                                                  mask_ints=mask_ints,
                                                  discriminator=discriminator,
                                                  pt_order=pt_order,
                                                  do_pt_order=do_pt_order)

    # Second pass: upper-triangle
    t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
    max_memory = getattr (las, 'max_memory', las.mol.max_memory)
    contracter = ContractHamCI (las, ints, nlas, lroots, h0, h1, h2, si_bra=si_bra,
                                si_ket=si_ket, mask_bra_space=mask_bra_space,
                                mask_ket_space=mask_ket_space, pt_order=pt_order,
                                do_pt_order=do_pt_order, add_transpose=add_transpose, accum=accum,
                                dtype=ci[0][0].dtype, max_memory=max_memory, log=log)
    lib.logger.timer (las, 'LASSI hci setup', *t0)
    hket_fr_pabq, t0 = contracter.kernel ()
    for i, hket_r_pabq in enumerate (hket_fr_pabq):
        for j, hket_pabq in enumerate (hket_r_pabq):
            if si_bra_is1d:
                hket_pabq = hket_pabq[0]
            if si_ket_is1d:
                hket_pabq = hket_pabq[:,:,:,:,0]
            hket_fr_pabq[i][j] = hket_pabq
    lib.logger.timer (las, 'LASSI hci crunching', *t0)
    if las.verbose >= lib.logger.TIMER_LEVEL:
        lib.logger.info (las, 'LASSI hci crunching profile:\n%s',
                         contracter.sprint_profile ())

    return hket_fr_pabq

