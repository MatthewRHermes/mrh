import numpy as np
from pyscf.lib import logger
from pyscf.lo.orth import vec_lowdin
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.fci.csfstring import CSFTransformer
from mrh.my_pyscf.fci.spin_op import contract_sdown, contract_sup
from mrh.my_pyscf.mcscf.lasci import get_space_info
from mrh.my_pyscf.mcscf.productstate import ProductStateFCISolver
from mrh.my_pyscf.lassi.excitations import ExcitationPSFCISolver
from mrh.my_pyscf.lassi.states import spin_shuffle, spin_shuffle_ci
from mrh.my_pyscf.lassi.states import all_single_excitations, SingleLASRootspace
from mrh.my_pyscf.lassi.lassi import LASSI

def prepare_states (lsi, nmax_charge=1, nmax_spin=0, sa_heff=True, deactivate_vrv=False, crash_locmin=False):
    las = lsi._las
    las1 = spin_shuffle (las, equal_weights=True)
    las1.ci = spin_shuffle_ci (las1, las1.ci)
    log = logger.new_logger (lsi, lsi.verbose)
    log.info ("LASSIS reference spaces: 0-%d", las1.nroots-1)
    for ix, (c, m, s, w) in enumerate (zip (*get_space_info (las1))):
        log.info ("Reference space %d:", ix)
        SingleLASRootspace (las1, m, s, c, 0).table_printlog ()
    # TODO: make states_energy_elec capable of handling lroots and address inconsistency
    # between definition of e_states array for neutral and charge-separated rootspaces
    las1.e_states = las1.energy_nuc () + np.array (las1.states_energy_elec ())
    if nmax_charge:
        las2 = all_single_excitations (las1)
        converged, las2.ci, las2.e_states = single_excitations_ci (
            lsi, las2, las1, nmax_charge=nmax_charge, sa_heff=sa_heff, deactivate_vrv=deactivate_vrv,
            crash_locmin=crash_locmin
        )
    else:
        converged, las2 = las1.converged, las1
    if lsi.nfrags > 3:
        raise NotImplementedError (">3-frag LASSIS")
        # TODO: direct product of single-electron hops
    if nmax_spin:
        raise NotImplementedError ("spin excitations")
        spins3, smults3, ci3 = single_fragment_spinsteps (lsi, las1, nmax_spin=nmax_spin)
        las3 = combine_charge_spin_excitations (las2, spins3, smults3, ci3)
    else:
        las3 = las2
    return converged, las3

def single_excitations_ci (lsi, las2, las1, nmax_charge=1, sa_heff=True, deactivate_vrv=False,
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
    spaces = [SingleLASRootspace (las2, m, s, c, las2.weights[ix], ci=[c[ix] for c in ci])
              for ix, (c, m, s, w) in enumerate (zip (*get_space_info (las2)))]
    ncsf = las2.get_ugg ().ncsf_sub
    if isinstance (nmax_charge, np.ndarray): nmax_charge=nmax_charge[None,:]
    lroots = np.minimum (nmax_charge, ncsf)
    h0, h1, h2 = lsi.ham_2q ()
    t0 = (logger.process_clock (), logger.perf_counter ())
    converged = True
    log.info ("LASSIS electron hop spaces: %d-%d", las1.nroots, las2.nroots-1)
    for i in range (las1.nroots, las2.nroots):
        psref = []
        ciref = [[] for j in range (nfrags)]
        excfrags = np.zeros (nfrags, dtype=bool)
        log.info ("Electron hop space %d:", i)
        spaces[i].table_printlog ()
        log.info ("is connected to reference spaces:")
        for j in range (las1.nroots):
            if not spaces[i].is_single_excitation_of (spaces[j]): continue
            src_frag = np.where ((spaces[i].nelec-spaces[j].nelec)==-1)[0][0]
            dest_frag = np.where ((spaces[i].nelec-spaces[j].nelec)==1)[0][0]
            e_spin = 'a' if np.any (spaces[i].neleca!=spaces[j].neleca) else 'b'
            src_ds = 'u' if spaces[i].smults[src_frag]>spaces[j].smults[src_frag] else 'd'
            dest_ds = 'u' if spaces[i].smults[dest_frag]>spaces[j].smults[dest_frag] else 'd'
            log.info ('%d: %d(%s) --%s--> %d(%s)', j, src_frag, src_ds, e_spin,
                      dest_frag, dest_ds)
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

def single_fragment_spinsteps (lsi, las, nmax_spin=1):
    norb_f = las.ncas_sub
    spaces = [SingleLASRootspace (las, m, s, c, las.weights[ix], ci=[c[ix] for c in las.ci])
              for ix, (c, m, s, w) in enumerate (zip (*get_space_info (las)))]
    if len (spaces) > 1:
        assert (all ([np.all(spaces[i].nelec==spaces[i-1].nelec) for i in range (1,len(spaces))]))
        assert (all ([np.all(spaces[i].smults==spaces[i-1].smults) for i in range (1,len(spaces))]))
    norb0 = las.ncas_sub
    nelec0 = spaces[0].nelec
    spins0 = spaces[0].spins
    smults0 = spaces[0].smults
    nfrags = spaces[0].nfrag
    smults1 = []
    spins1 = []
    ci1 = []
    h0, h1, h2 = lsi.ham_2q ()
    casdm1s = las.make_casdm1s ()
    f1 = h1 + np.tensordot (h2, casdm1s.sum (1), axes=2)
    f1 = f1[None,:,:] - np.tensordot (casdm1s, h2, axes=((1,2),(2,1)))
    i = 0
    for ifrag, (norb, nelec, spin, smult) in enumerate (zip (norb0, nelec0, spins0, smults0)):
        j = i + norb
        h2_i = h2[i:j,i:j,i:j,i:j]
        lasdm1s = casdm1s[:,i:j,i:j]
        h1_i = (f1[:,i:j,i:j] - np.tensordot (h2_i, lasdm1s.sum (0))[None,:,:]
                + np.tensordot (lasdm1s, h2_i, axes=((1,2),(2,1))))
        def cisolve (sm):
            neleca = (nelec + (sm-1)) // 2
            nelecb = (nelec - (sm-1)) // 2
            solver = csf_solver (las.mol, smult=sm).set (nelec=(neleca,nelecb), norb=norb)
            solver.check_transformer_cache ()
            nroots = min (nmax_spin, solver.transformer.ncsf)
            ci_list = solver.kernel (h1_i, h2_i, norb, (neleca,nelecb), nroots=nroots)[1]
            if nroots==1: ci_list = [ci_list,]
            ci_arrlist = [np.array (ci_list),]
            for ms in range (1,sm):
                ci_list = [contract_sdown (ci) for ci in ci_list]
                ci_arrlist.append (np.array (ci_list))
            return ci_arrlist
        smults1_i = []
        spins1_i = []
        ci1_i = []
        if smult > 2: # spin-lowered
            smults1_i.extend ([smult-2,]*(smult-2))
            spins1_i.extend (list (range (smult-1, -(smult-1), -2)))
            ci1_i.extend (cisolve (smult-2))
        min_npair = max (0, nelec-norb)
        max_smult = (nelec - 2*min_npair) + 1
        if smult < max_smult: # spin-raised
            smults1_i.extend ([smult+2,]*(smult+2))
            spins1_i.extend (list (range (smult+1, -(smult+1), -2)))
            ci1_i.extend (cisolve (smult+2))
        smults1.append (smults1_i)
        spins1.append (spins1_i)
        ci1.append (ci1_i)
        i = j
    return spins1, smults1, ci1

def combine_charge_spin_excitations (las2, spins3, smults3, ci3):
    spaces = [SingleLASRootspace (las2, m, s, c, las2.weights[ix], ci=[c[ix] for c in las2.ci])
              for ix, (c, m, s, w) in enumerate (zip (*get_space_info (las2)))]
    nelec0 = spaces[0].nelec
    smults0 = spaces[0].smults
    nfrags = spaces[0].nfrag
    spin = spaces[0].spins.sum ()
    for ifrag in range (nfrags):
        new_spaces = []
        m3, s3, c3 = spins3[ifrag], smults3[ifrag], ci3[ifrag]
        for space in spaces:
            if space.nelec[ifrag] != nelec0[ifrag]: continue
            if space.smults[ifrag] != smults0[ifrag]: continue
            for m3i, s3i, c3i in zip (m3, s3, c3):
                new_spaces.append (space.single_fragment_spin_change (
                    ifrag, s3, m3, ci=c3))
        spaces += new_spaces
    spaces = [space for space in new_spaces if space.spins.sum () == spin]
    weights = [space.weight for space in spaces]
    charges = [space.charges for space in spaces]
    spins = [space.spins for space in spaces]
    smults = [space.smults for space in spaces]
    ci3 = [[space.ci[ifrag] for space in spaces] for ifrag in range (nfrags)]
    las3 = las2.state_average (weights=weights, charges=charges, spins=spins, smults=smults)
    las3.ci = ci3
    return las3

class LASSIS (LASSI):
    def __init__(self, las, nmax_charge=1, nmax_spin=0, sa_heff=True, deactivate_vrv=False,
                 crash_locmin=False, **kwargs):
        self.nmax_charge = nmax_charge
        self.nmax_spin = nmax_spin
        self.sa_heff = sa_heff
        self.deactivate_vrv = deactivate_vrv
        self.crash_locmin = crash_locmin
        self.e_states_meaningless = True # a tag to silence an invalid warning
        LASSI.__init__(self, las, **kwargs)
        if las.nroots>1:
            logger.warn (self, ("LASSIS builds the model space for you! I don't know what will "
                                "happen if you build a model space by hand!"))
    def kernel (self, nmax_charge=None, nmax_spin=None, sa_heff=None, deactivate_vrv=None,
                crash_locmin=None, **kwargs):
        if nmax_charge is None: nmax_charge = self.nmax_charge
        if nmax_spin is None: nmax_spin = self.nmax_spin
        if sa_heff is None: sa_heff = self.sa_heff
        if deactivate_vrv is None: deactivate_vrv = self.deactivate_vrv
        if crash_locmin is None: crash_locmin = self.crash_locmin
        self.converged, las = prepare_states (self, nmax_charge=nmax_charge, nmax_spin=nmax_spin,
                                              sa_heff=sa_heff, deactivate_vrv=deactivate_vrv,
                                              crash_locmin=crash_locmin)
        self.__dict__.update(las.__dict__)
        return LASSI.kernel (self, **kwargs)

