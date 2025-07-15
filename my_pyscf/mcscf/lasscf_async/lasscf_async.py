import itertools
import numpy as np
from scipy import linalg
from pyscf import lib
from pyscf.mcscf import mc1step
from mrh.my_pyscf.mcscf import lasci, lasscf_sync_o0
from mrh.my_pyscf.mcscf.lasscf_guess import interpret_frags_atoms
from mrh.my_pyscf.mcscf.lasscf_async import keyframe, combine
from mrh.my_pyscf.mcscf.lasscf_async.split import get_impurity_space_constructor
from mrh.my_pyscf.mcscf.lasscf_async.crunch import get_impurity_casscf

def kernel (las, mo_coeff=None, ci0=None, conv_tol_grad=1e-4,
            assert_no_dupes=False, verbose=lib.logger.NOTE, frags_orbs=None,
            **kwargs):
    t_setup = (lib.logger.process_clock(), lib.logger.perf_counter())    
    if mo_coeff is None: mo_coeff = las.mo_coeff
    if assert_no_dupes: las.assert_no_duplicates ()
    h2eff_sub = las.get_h2eff (mo_coeff)
    if (ci0 is None or any ([c is None for c in ci0]) or
      any ([any ([c2 is None for c2 in c1]) for c1 in ci0])):
        ci0 = las.get_init_guess_ci (mo_coeff, h2eff_sub, ci0)
    if (ci0 is None or any ([c is None for c in ci0]) or
      any ([any ([c2 is None for c2 in c1]) for c1 in ci0])):
        raise RuntimeError ("failed to populate get_init_guess")
    if frags_orbs is None: frags_orbs = getattr (las, 'frags_orbs', None)
    imporb_builders = [get_impurity_space_constructor (las, i, frag_orbs=frag_orbs)
                       for i, frag_orbs in enumerate (frags_orbs)]
    nfrags = len (las.ncas_sub)
    log = lib.logger.new_logger(las, verbose)
    t0 = (lib.logger.process_clock(), lib.logger.perf_counter())
    kf0 = las.get_keyframe (mo_coeff, ci0) 
    las._flas_stdout = {} # TODO: more elegant model for this
    ugg = las.get_ugg ()

    e_tot = las.energy_nuc () + las.energy_elec (
        mo_coeff=kf0.mo_coeff, ci=kf0.ci, h2eff=kf0.h2eff_sub, veff=kf0.veff)
    gvec = las.get_grad (ugg=ugg, kf=kf0)
    norm_gvec = linalg.norm (gvec)
    log.info ('LASSCF macro 0 : E = %.15g ; |g| = %.15g', e_tot, norm_gvec)
    t_setup = log.timer( "LASSCF setup", *t_setup)

    ###############################################################################################
    ################################## Begin actual kernel logic ##################################
    ###############################################################################################





    converged = False
    it = 0
    kf1 = kf0
    impurities = [get_impurity_casscf (las, i, imporb_builder=builder)
                  for i, builder in enumerate (imporb_builders)]
    t1 = log.timer_debug1 ('impurity solver construction', *t0)
    # GRAND CHALLENGE: replace rigid algorithm below with dynamic task scheduling
    for it in range (las.max_cycle_macro):
        t_macro = (lib.logger.process_clock(), lib.logger.perf_counter())    
        # 1. Divide into fragments
        for impurity in impurities: 
            impurity._pull_keyframe_(kf1)
            t_macro = log.timer("Pull keyframe for fragment",*t_macro)
        
        # 2. CASSCF on each fragment
        kf2_list = []
        for impurity in impurities:
            impurity.kernel ()
            t_macro = log.timer("Fragment CASSCF",*t_macro)
            kf2_list.append (impurity._push_keyframe (kf1))
            t_macro = log.timer("Push keyframe for fragment",*t_macro)

            
        # 3. Combine from fragments. It should not be necessary to do this in any particular order,
        #    and the below does it March Madness tournament style; e.g.:
        #
        #       kf2_list[0] --- kf2_list[1]     kf2_list[2] --- kf2_list[3]
        #                    |                               |
        #                   kfi --------------------------- kfj
        #                                    |
        #                                   kf2
        #
        nkf = len (kf2_list)
        ncyc = int (np.ceil (np.log2 (nkf)))
        for i in range (int (np.ceil (np.log2 (nkf)))):
            nkfi = len (kf2_list)
            kf3_list = []
            for kf2, kf3 in zip (kf2_list[::2],kf2_list[1::2]):
                kf3_list.append (combine.combine_pair (las, kf2, kf3, kf_ref=kf1))
                t_macro = log.timer("Recombination",*t_macro)
            if nkfi%2: kf3_list.insert (len(kf3_list)-1, kf2_list[-1])
            # Insert this at second-to-last position so that it gets "mixed in" next cycle
            kf2_list = kf3_list
        assert (len (kf2_list) == 1)
        kf1 = kf2_list[0]

        # Evaluate status and break if converged
        e_tot = las.energy_nuc () + las.energy_elec (
            mo_coeff=kf1.mo_coeff, ci=kf1.ci, h2eff=kf1.h2eff_sub, veff=kf1.veff)
        gvec = las.get_grad (ugg=ugg, kf=kf1)
        norm_gvec = linalg.norm (gvec)
        log.info ('LASSCF macro %d : E = %.15g ; |g| = %.15g', it+1, e_tot, norm_gvec)
        if verbose > lib.logger.INFO: keyframe.gradient_analysis (las, kf1, log)
        t1 = log.timer ('one LASSCF macro cycle', *t1)
        las.dump_chk (mo_coeff=kf1.mo_coeff, ci=kf1.ci)
        if norm_gvec < conv_tol_grad:
            converged = True
            break





    ###############################################################################################
    ################################### End actual kernel logic ###################################
    ###############################################################################################

    for key, val in las._flas_stdout.items (): val.close ()
    # TODO: more elegant model for this
    mo_coeff, ci1, h2eff_sub, veff = kf1.mo_coeff, kf1.ci, kf1.h2eff_sub, kf1.veff
    t1 = log.timer ('LASSCF {} macrocycles'.format (it), *t0)
    e_tot = las.energy_nuc () + las.energy_elec (mo_coeff=mo_coeff, ci=ci1, h2eff=h2eff_sub,
                                                 veff=veff)
    e_states = las.energy_nuc () + np.array (las.states_energy_elec (mo_coeff=mo_coeff, ci=ci1,
                                                                     h2eff=h2eff_sub, veff=veff))
    # This crap usually goes in a "_finalize" function
    log.info ('LASSCF %s after %d cycles', ('not converged', 'converged')[converged], it+1)
    log.info ('LASSCF E = %.15g ; |g| = %.15g', e_tot,
              norm_gvec)
    t1 = log.timer ('LASSCF final energy', *t1)
    mo_coeff, mo_energy, mo_occ, ci1, h2eff_sub = las.canonicalize (mo_coeff, ci1, veff=veff,
                                                                    h2eff_sub=h2eff_sub)
    t1 = log.timer ('LASSCF canonicalization', *t1)
    t0 = log.timer ('LASSCF kernel function', *t0)

    e_cas = None # TODO: get rid of this worthless, meaningless variable
    return converged, e_tot, e_states, mo_energy, mo_coeff, e_cas, ci1, h2eff_sub, veff

def get_grad (las, mo_coeff=None, ci=None, ugg=None, kf=None):
    '''Return energy gradient for orbital rotation and CI relaxation.

    Args:
        las : instance of :class:`LASCINoSymm`

    Kwargs:
        mo_coeff : ndarray of shape (nao,nmo)
            Contains molecular orbitals
        ci : list (length=nfrags) of list (length=nroots) of ndarray
            Contains CI vectors
        ugg : instance of :class:`LASCI_UnitaryGroupGenerators`
        kf : instance of :class:`LASKeyframe`
            Overrides mo_coeff and ci if provided and carries other intermediate
            quantities that may have been calculated in advance

    Returns:
        gvec : ndarray of shape (ugg.nvar_tot)
            Contains collapsed 1d gradient
    '''
    if mo_coeff is None: mo_coeff=las.mo_coeff
    if ci is None: ci=las.ci
    if ugg is None: ugg=las.get_ugg ()
    if kf is None: kf=las.get_keyframe (mo_coeff, ci)
    mo_coeff, ci = kf.mo_coeff, kf.ci
    veff, fock1 = kf.veff, kf.fock1
    h2eff_sub, h1eff_sub = kf.h2eff_sub, kf.h1eff_sub

    gorb = fock1 - fock1.T
    gci = las.get_grad_ci (mo_coeff=mo_coeff, ci=ci, h1eff_sub=h1eff_sub, h2eff_sub=h2eff_sub,
                           veff=veff)
    return ugg.pack (gorb, gci)

class SortedIndexDict (dict):
    '''A dict, but all keys that are tuples are sorted so that, for instance, (1,2) is always
    the same as (2,1)'''
    def __setitem__(self, key, val):
        if isinstance (key, tuple): key = tuple (sorted (key))
        dict.__setitem__(self, key, val)
    def __getitem__(self, key):
        if isinstance (key, tuple): key = tuple (sorted (key))
        return dict.__getitem__(self, key)
    def get (self, key, *args):
        if isinstance (key, tuple): key = tuple (sorted (key))
        if len (args):
            return dict.get (self, key, *args)
        else:
            return dict.get (self, key)

class LASSCFNoSymm (lasci.LASCINoSymm):
    '''Extra attributes:

    frags_orbs : list of length nfrags of list of integers
        Identifies the definition of fragments as lists of AOs
    impurity_params : list of length nfrags of dict
        Key/value pairs are assigned as attributes of the impurity solver CASSCF object.
        Use this to address, e.g., conv_tol_grad, max_cycle_macro, etc. of the impurity
        subproblems
    relax_params : dict
        Key/value pairs are assigned as attributes to the active-active relaxation (``LASCI'')
        subproblem, similar to impurity_params. Use this to, e.g., set a different max_cycle_macro
        for the ``LASCI'' step.
    combine_pair_max_frags : integer
        Maximum number of frags to simultaneously relax during the combine_pair step.
    '''
    def __init__(self, mf, ncas, nelecas, ncore=None, spin_sub=None, **kwargs):
        lasci.LASCINoSymm.__init__(self, mf, ncas, nelecas, ncore=ncore, spin_sub=spin_sub,
                                   **kwargs)
        self.impurity_params = {}
        for i in range (self.nfrags):
            self.impurity_params[i] = {}
        self.relax_params = {}
        for i, j in itertools.combinations (range (self.nfrags), 2):
            self.relax_params[(i,j)] = {}
        self.combine_pair_max_frags = self.nfrags
        keys = set (('frags_orbs','impurity_params','relax_params','combine_pair_max_frags'))
        self._keys = self._keys.union (keys)

    @property
    def relax_params (self): return self._relax_params
    @relax_params.setter
    def relax_params (self, d):
        self._relax_params = SortedIndexDict ()
        for key, val in d.items ():
            self._relax_params[key] = val

    _ugg = lasscf_sync_o0.LASSCF_UnitaryGroupGenerators
    _kern = kernel
    get_grad = get_grad
    def get_keyframe (self, mo_coeff=None, ci=None):
        if mo_coeff is None: mo_coeff=self.mo_coeff
        if ci is None: ci=self.ci
        return keyframe.LASKeyframe (self, mo_coeff, ci)
    as_scanner = mc1step.as_scanner
    def set_fragments_(self, frags_atoms=None, mo_coeff=None, localize_init_guess=True,
                       frags_by_AOs=False, **kwargs):
        ''' Project active orbitals into sets of orthonormal "fragments" defined by frags_atoms,
        cache the resulting frags_orbs member, and call localize_init_guess. Passes on unlisted
        kwargs to localize_init_guess call.
    
        Kwargs:
            frags_atoms: list of length nfrags
                Contains either lists of integer atom indices, or lists of
                strings which are passed to mol.search_ao_label, which define
                fragments into which the active orbitals are to be localized
            mo_coeff: ndarray of shape (nao, nmo)
                Molecular orbital coefficients containing active orbitals
                on columns ncore:ncore+ncas
            localize_init_guess: logical
                If false, this step is skipped and mo_coeff is returned unaltered
            frags_by_AOs: logical
                If True, interpret integer frags_atoms as AOs rather than atoms
    
        Returns:
            mo_coeff: ndarray of shape (nao,nmo)
                Orbital coefficients after localization of the active space;
                columns in the order (inactive,las1,las2,...,lasn,external)
        '''
        self.frags_orbs = interpret_frags_atoms (self.mol, frags_atoms, frags_by_AOs=frags_by_AOs)
        if mo_coeff is None: mo_coeff=self.mo_coeff
        if localize_init_guess:
            mo_coeff = self.localize_init_guess (self.frags_orbs, mo_coeff=mo_coeff,
                                                 frags_by_AOs=True, **kwargs) 
        return mo_coeff
    def dump_flags (self, verbose=None, _method_name='LASSCF'):
        lasci.LASCINoSymm.dump_flags (self, verbose=verbose, _method_name=_method_name)
    def _finalize(self):
        log = lib.logger.new_logger (self, self.verbose)
        nroots_prt = len (self.e_states)
        if self.verbose <= lib.logger.INFO:
            nroots_prt = min (nroots_prt, 100)
        if nroots_prt < len (self.e_states):
            log.info (("Printing a maximum of 100 state energies;"
                       " increase self.verbose to see them all"))
        if nroots_prt > 1:
            log.info ("LASSCF state-average energy = %.15g", self.e_tot)
            for i, e in enumerate (self.e_states):
                log.info ("LASSCF state %d energy = %.15g", i, e)
        else:
            log.info ("LASSCF energy = %.15g", self.e_tot)
        return

class LASSCFSymm (lasci.LASCISymm):
    def __init__(self, mf, ncas, nelecas, ncore=None, spin_sub=None, **kwargs):
        lasci.LASCISymm.__init__(self, mf, ncas, nelecas, ncore=ncore, spin_sub=spin_sub, **kwargs)
        self.impurity_params = [{} for i in range (self.nfrags)]
        self.relax_params = {}
        keys = set (('frags_orbs','impurity_params','relax_params'))
        self._keys = self._keys.union (keys)

    _ugg = lasscf_sync_o0.LASSCFSymm_UnitaryGroupGenerators
    _kern = kernel
    _finalize = LASSCFNoSymm._finalize
    get_grad = get_grad
    get_keyframe = LASSCFNoSymm.get_keyframe
    as_scanner = mc1step.as_scanner
    set_fragments_ = LASSCFNoSymm.set_fragments_
    dump_flags = LASSCFNoSymm.dump_flags

def LASSCF (mf_or_mol, ncas_sub, nelecas_sub, **kwargs):
    # try grabbing gpu handle from mf_or_mol instead of additional argument
    use_gpu = kwargs.get('use_gpu', None)
    
    from pyscf import gto, scf
    if isinstance(mf_or_mol, gto.Mole):
        mf = scf.RHF(mf_or_mol)
    elif isinstance (mf_or_mol, scf.hf.SCF):
        mf = mf_or_mol
    else:
        raise RuntimeError ("LASSCF constructor requires molecule or SCF instance")
    if mf.mol.symmetry:
        las = LASSCFSymm (mf, ncas_sub, nelecas_sub, **kwargs)
    else:
        las = LASSCFNoSymm (mf, ncas_sub, nelecas_sub, **kwargs)
    if getattr (mf, 'with_df', None):
        las = lasci.density_fit (las, with_df = mf.with_df)
    return las

if __name__=='__main__':
    from mrh.tests.lasscf.c2h6n4_struct import structure as struct
    from pyscf import scf
    mol = struct (1.0, 1.0, '6-31g', symmetry=False)
    mol.verbose = 5
    mol.output = 'lasscf_async.log'
    mol.build ()
    mf = scf.RHF (mol).run ()
    las = LASSCF (mf, (4,4), ((4,0),(0,4)), spin_sub=(5,5))
    mo = las.set_fragments_((list (range (3)), list (range (9,12))), mf.mo_coeff)
    las.state_average_(weights=[1,0,0,0,0],
                       spins=[[0,0],[2,0],[-2,0],[0,2],[0,-2]],
                       smults=[[1,1],[3,1],[3,1],[1,3],[1,3]])
    las.kernel (mo)


