import numpy as np
from pyscf.lib import logger
from pyscf.lo.orth import vec_lowdin
from mrh.my_pyscf.mcscf.lasci import get_space_info
from mrh.my_pyscf.mcscf.productstate import ProductStateFCISolver
from mrh.my_pyscf.lassi.excitations import ExcitationPSFCISolver
from mrh.my_pyscf.lassi.states import spin_shuffle, spin_shuffle_ci
from mrh.my_pyscf.lassi.states import all_single_excitations, SingleLASRootspace
from mrh.my_pyscf.lassi.lassi import LASSI

def prepare_states (lsi, nmax_charge=0, sa_heff=True, deactivate_vrv=False, crash_locmin=False):
    las = lsi._las
    las1 = spin_shuffle (las)
    las1.ci = spin_shuffle_ci (las1, las1.ci)
    # TODO: make states_energy_elec capable of handling lroots and address inconsistency
    # between definition of e_states array for neutral and charge-separated rootspaces
    las1.e_states = las1.energy_nuc () + np.array (las1.states_energy_elec ())
    las2 = all_single_excitations (las1)
    converged, las2.ci, las2.e_states = single_excitations_ci (
        lsi, las2, las1, nmax_charge=nmax_charge, sa_heff=sa_heff, deactivate_vrv=deactivate_vrv,
        crash_locmin=crash_locmin
    )
    return converged, las2

def single_excitations_ci (lsi, las2, las1, nmax_charge=0, sa_heff=True, deactivate_vrv=False,
                           crash_locmin=False):
    log = logger.new_logger (lsi, lsi.verbose)
    mol = lsi.mol
    nfrags = lsi.nfrags
    e_roots = np.append (las1.e_states, np.zeros (las2.nroots-las1.nroots))
    psrefs = []
    ci = [[ci_ij for ci_ij in ci_i] for ci_i in las2.ci]
    for j in range (las1.nroots):
        solvers = [b.fcisolvers[j] for b in las1.fciboxes]
        psrefs.append (ProductStateFCISolver (solvers, stdout=mol.stdout, verbose=mol.verbose))
    spaces = [SingleLASRootspace (las2, m, s, c, 0, ci=[c[ix] for c in ci])
              for ix, (c, m, s, w) in enumerate (zip (*get_space_info (las2)))]
    ncsf = las2.get_ugg ().ncsf_sub
    if isinstance (nmax_charge, np.ndarray): nmax_charge=nmax_charge[None,:]
    lroots = np.minimum (1+nmax_charge, ncsf)
    h0, h1, h2 = lsi.ham_2q ()
    t0 = (logger.process_clock (), logger.perf_counter ())
    converged = True
    for i in range (las1.nroots, las2.nroots):
        psref = []
        ciref = [[] for j in range (nfrags)]
        excfrags = np.zeros (nfrags, dtype=bool)
        for j in range (las1.nroots):
            if not spaces[i].is_single_excitation_of (spaces[j]): continue
            excfrags[spaces[i].excited_fragments (spaces[j])] = True
            psref.append (psrefs[j])
            for k in range (nfrags):
                ciref[k].append (las1.ci[k][j])
        psexc = ExcitationPSFCISolver (psref, ciref, las2.ncas_sub, las2.nelecas_sub,
                                       stdout=mol.stdout, verbose=mol.verbose,
                                       crash_locmin=crash_locmin)
        psexc._deactivate_vrv = deactivate_vrv
        neleca = spaces[i].neleca
        nelecb = spaces[i].nelecb
        smults = spaces[i].smults
        for k in np.where (excfrags)[0]:
            weights = np.zeros (lroots[k,i])
            if sa_heff: weights[:] = 1.0 / len (weights)
            else: weights[0] = 1.0
            psexc.set_excited_fragment_(k, (neleca[k],nelecb[k]), smults[k], weights=weights)
        conv, e_roots[i], ci1 = psexc.kernel (h1, h2, ecore=h0,
                                              max_cycle_macro=lsi.max_cycle_macro,
                                              conv_tol_self=1)
        spin_shuffle_ref = all ([spaces[j].is_spin_shuffle_of (spaces[0])
                                 for j in range (1,las1.nroots)])
        for k in np.where (~excfrags)[0]:
            # ci vector shape issues
            if las1.nroots==1:
                ci1[k] = np.asarray (ci1[k])
            elif spin_shuffle_ref:
                # NOTE: This logic fails if the user does spin_shuffle -> lasci -> LASSIS
                ci1[k] = np.asarray (ci1[k][0])
            else:
                ndeta, ndetb = space[i].get_ndet (k)
                ci1[k] = np.concatenate ([c.reshape (-1,ndeta,ndetb) for c in ci1[k]], axis=0)
                ci1[k] = vec_lowdin (ci1[k])
        for k in range (nfrags):
            if isinstance (ci1[k], list):
                print (k, len (ci1[k]), np.asarray (ci1[k]).shape, type (ciref[k]), [c.shape for c in ciref[k]])
        if not conv: log.warn ("CI vectors for charge-separated rootspace %d not converged", i)
        converged = converged and conv
        for k in range (nfrags):
            ci[k][i] = ci1[k]
        t0 = log.timer ("Space {} excitations".format (i), *t0)
    return converged, ci, e_roots

class LASSIS (LASSI):
    def __init__(self, las, nmax_charge=0, sa_heff=True, deactivate_vrv=False, crash_locmin=False, 
                 **kwargs):
        self.nmax_charge = nmax_charge
        self.sa_heff = sa_heff
        self.deactivate_vrv = deactivate_vrv
        self.crash_locmin = crash_locmin
        self.e_states_meaningless = True # a tag to silence an invalid warning
        LASSI.__init__(self, las, **kwargs)
        if las.nroots>1:
            logger.warn (self, ("LASSIS builds the model space for you! I don't know what will "
                                "happen if you build a model space by hand!"))
    def kernel (self, nmax_charge=None, sa_heff=None, deactivate_vrv=None, crash_locmin=None, 
                **kwargs):
        if nmax_charge is None: nmax_charge = self.nmax_charge
        if sa_heff is None: sa_heff = self.sa_heff
        if deactivate_vrv is None: deactivate_vrv = self.deactivate_vrv
        if crash_locmin is None: crash_locmin = self.crash_locmin
        self.converged, las = prepare_states (self, nmax_charge=nmax_charge, sa_heff=sa_heff,
                                              deactivate_vrv=deactivate_vrv,
                                              crash_locmin=crash_locmin)
        self.__dict__.update(las.__dict__)
        return LASSI.kernel (self, **kwargs)

