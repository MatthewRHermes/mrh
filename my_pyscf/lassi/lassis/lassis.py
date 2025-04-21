import sys
import numpy as np
import itertools
from scipy import linalg
from pyscf import lib, gto
from pyscf.lib import logger
from pyscf.lo.orth import vec_lowdin
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.fci.csfstring import CSFTransformer
from mrh.my_pyscf.fci.spin_op import contract_sdown, contract_sup, mdown, mup
from mrh.my_pyscf.mcscf.lasci import get_space_info
from mrh.my_pyscf.mcscf.productstate import ProductStateFCISolver
from mrh.my_pyscf.lassi.lassis.excitations import ExcitationPSFCISolver
from mrh.my_pyscf.lassi.spaces import spin_shuffle, spin_shuffle_ci
from mrh.my_pyscf.lassi.spaces import _spin_shuffle, list_spaces
from mrh.my_pyscf.lassi.spaces import all_single_excitations
from mrh.my_pyscf.lassi.spaces import orthogonal_excitations, combine_orthogonal_excitations
from mrh.my_pyscf.lassi.lassi import LASSI

# TODO: split prepare_states into three steps
# 1. Compute the number of unique fragment CI vectors to be computed (including sz-flips but not
#    including repeated references in the final CI table), the total number of model states in the
#    final step, and report the memory footprint. Prepare the corresponding tables of CI vectors.
# 2. Optimize the unique, unfrozen CI vectors. Include the option to initialize them from stored
#    values.
# 3. Combine the optimized CI vectors into a single ci table in the format the LASSI kernel expects.
#    Use references, not copies.

def prepare_model_states (lsi, ci_ref, ci_sf, ci_ch):
    t0 = (logger.process_clock (), logger.perf_counter ())
    log = logger.new_logger (lsi, lsi.verbose)
    las = lsi.get_las_of_ci_ref (ci_ref)
    space0 = list_spaces (las)[0]
    # Make spin flip objects
    spin_flips = []
    for i in range (las.nfrags):
        smult = space0.smults[i]
        ci1 = []
        spins1 = []
        smults1 = []
        if ci_sf[i][0] is not None:
            smults1.append (smult-2)
            spins1.append (smult-3)
            ci1.append (ci_sf[i][0])
        if ci_sf[i][1] is not None:
            smults1.append (smult+2)
            spins1.append (smult+1)
            ci1.append (ci_sf[i][1])
        spin_flips.append (SpinFlips (ci1, space0.nlas[i], space0.nelec[i], spins1, smults1))
    # Make charge-hop objects
    spaces = [space0]
    for i, a in itertools.product (range (lsi.nfrags), repeat=2):
        for s in range (4):
            ci_i, ci_a = ci_ch[i][a][s]
            if ci_i is None or ci_a is None: continue
            dsi = -1 + (s//2)*2
            dsa = -1 + (s%2)*2
            spaces.append (space0.get_single_any_m (i, a, dsi, dsa, ci_i=ci_i, ci_a=ci_a))
    # Excitation products and spin-shuffling
    if lsi.nfrags > 3:
        spaces = charge_excitation_products (lsi, spaces, nroots_ref=1)
    spaces = spin_flip_products (las, spaces, spin_flips, nroots_ref=1)
    # Throat-clear
    weights = [space.weight for space in spaces]
    charges = [space.charges for space in spaces]
    spins = [space.spins for space in spaces]
    smults = [space.smults for space in spaces]
    ci = [[space.ci[ifrag] for space in spaces] for ifrag in range (lsi.nfrags)]
    entmaps = [space.entmap for space in spaces]
    las = las.state_average (weights=weights, charges=charges, spins=spins, smults=smults, assert_no_dupes=False)
    las.ci = ci
    las.lasci (_dry_run=True)
    log.timer ("LASSIS model space preparation", *t0)
    return las, entmaps

def prepare_fbf (lsi, ci_ref, ci_sf, ci_ch, ncharge=1, nspin=0, sa_heff=True,
                 deactivate_vrv=False, crash_locmin=False):
    t0 = (logger.process_clock (), logger.perf_counter ())
    ham_2q = lsi.ham_2q ()
    log = logger.new_logger (lsi, lsi.verbose)
    las = lsi.get_las_of_ci_ref (ci_ref)
    # 1. Spin shuffle step
    if np.all (get_space_info (las)[2]==1):
        # If all singlets, skip the spin shuffle and the unnecessary warning below
        las1 = las
    else:
        las1 = spin_shuffle (las, equal_weights=True)
        # TODO: memory efficiency; the line below makes copies
        las1.ci = spin_shuffle_ci (las1, las1.ci)
        las1.converged = las.converged
    nroots_ref = las1.nroots
    # 2. Spin excitations part 1
    spin_flips, conv_sf = None, True
    if nspin: conv_sf,spin_flips,ci_sf = all_spin_flips (lsi,las1,ci_sf,nspin=nspin,ham_2q=ham_2q)
    las1.e_states = las1.energy_nuc () + np.array (las1.states_energy_elec ())
    # 3. Charge excitations
    if ncharge:
        las2 = all_single_excitations (las1)
        conv_ch, ci_ch, max_disc_sval = single_excitations_ci (
            lsi, las2, las1, ci_ch, ncharge=ncharge, sa_heff=sa_heff,
            deactivate_vrv=deactivate_vrv, spin_flips=spin_flips, crash_locmin=crash_locmin,
            ham_2q=ham_2q
        )
    log.timer ("LASSIS fragment basis functions preparation", *t0)
    return conv_sf and conv_ch, ci_sf, ci_ch, max_disc_sval

def filter_single_excitation_spin_shuffles (lsi, spaces, nroots_ref=1):
    spaces_ref = spaces[:nroots_ref]
    spaces = spaces[nroots_ref:]
    space0 = spaces_ref[0]

    manifolds = []
    for space in spaces:
        isnew = True
        for manifold in manifolds:
            if space.is_spin_shuffle_of (manifold[0]):
                manifold.append (space)
                isnew = False
                break
        if isnew:
            manifold = [space,]
            manifolds.append (manifold)

    spaces = [select_single_excitation_from_spin_manifold (lsi, space0, manifold)
              for manifold in manifolds]
    return spaces_ref + spaces

def select_single_excitation_from_spin_manifold (lsi, space0, manifold):
    log = lib.logger.new_logger (lsi)
    nelec0 = space0.nelec
    smults0 = space0.smults
    spins0 = space0.spins
    nelec1 = manifold[0].nelec
    smults1 = manifold[0].smults
    ifrag = np.where ((nelec1-nelec0)==-1)[0][0]
    afrag = np.where ((nelec1-nelec0)==1)[0][0]
    spins1 = np.abs (spins0.copy ())
    target_sign = np.sign (spins0)
    if (spins0[ifrag] == 0) and (spins0[afrag] == 0):
        # arbitrarily preference alpha-electron hopping
        target_sign[ifrag] = -1
        target_sign[afrag] = 1
    elif spins0[ifrag] == 0:
        # set preference by receiving fragment
        target_sign[ifrag] = -target_sign[afrag]
    elif spins0[afrag] == 0:
        # set preference by donating fragment
        target_sign[afrag] = -target_sign[ifrag]
    spins1[ifrag] += smults1[ifrag]-smults0[ifrag]
    spins1[afrag] += smults1[afrag]-smults0[afrag]
    spins1 = target_sign * spins1
    assert (np.all (np.abs (spins1) < smults1))
    spins = np.stack ([space.spins for space in manifold], axis=0)
    dspins = np.abs (spins - spins1[None,:])
    # Sort by smults; first for environment, then for active frags
    sorter = smults1.copy ()
    offset = np.amax (sorter)
    sorter[ifrag] += offset
    sorter[afrag] += offset
    if sorter[ifrag] == sorter[afrag]: 
        sorter[afrag] += 1
    idx = np.argsort (sorter, kind='stable')
    dspins = dspins[:,idx]
    dimsize = dspins.shape[0] * np.amax (dspins, axis=0)
    dimsize = np.cumprod (dimsize[::-1]+1)[::-1]
    scores = np.dot (dimsize, dspins.T)
    # debrief
    logstr = 'excitation {}->{}\nnelec_ref: {}\nsmults_ref: {}\nsmults_exc: {}\n'.format (
        ifrag, afrag, nelec0, smults0, smults1)
    logstr += 'ref spins: {}\ntarget spins: {}\n'.format (spins0, spins1)
    for i, space in enumerate (manifold):
        logstr += 'candidate spins: {}, score: {}\n'.format (space.spins, scores[i])
    log.debug (logstr)
    idx = (scores == np.amin (scores))
    manifold = [space for i, space in enumerate (manifold) if idx[i]]
    if len (manifold) > 1: raise RuntimeError (logstr)
    return manifold[0]

def single_excitations_ci (lsi, las2, las1, ci_ch, ncharge=1, sa_heff=True, deactivate_vrv=False,
                           spin_flips=None, crash_locmin=False, ham_2q=None):
    log = logger.new_logger (lsi, lsi.verbose)
    mol = lsi.mol
    nfrags = lsi.nfrags
    e_roots = np.append (las1.e_states, np.zeros (las2.nroots-las1.nroots))
    spaces = list_spaces (las2)
    ncsf = las2.get_ugg ().ncsf_sub
    auto_singles = False
    if isinstance (ncharge, np.ndarray):
        ncharge=ncharge[None,:]
    elif isinstance (ncharge, str):
        if 's' in ncharge.lower ():
            auto_singles = True
            ncharge = ncsf
        else:
            raise RuntimeError ("Valid ncharge values are integers or 's'")
    lroots = np.minimum (ncharge, ncsf)
    if ham_2q is None:
        h0, h1, h2 = lsi.ham_2q ()
    else:
        h0, h1, h2 = ham_2q
    t0 = (logger.process_clock (), logger.perf_counter ())
    converged = True
    spaces = filter_single_excitation_spin_shuffles (lsi, spaces, nroots_ref=las1.nroots)
    keys = set ()
    max_max_disc = 0
    for i in range (las1.nroots, len (spaces)):
        # compute lroots
        psref_ix = [j for j, space in enumerate (spaces[:las1.nroots])
                    if spaces[i].is_single_excitation_of (space)]
        psref = [spaces[j] for j in psref_ix]
        excfrags = np.zeros (nfrags, dtype=bool)
        for space in psref: excfrags[spaces[i].excited_fragments (space)] = True
        nref_pure = len (psref)
        psref = _spin_flip_products (psref, spin_flips, nroots_ref=len(psref),
                                               frozen_frags=(~excfrags))
        psref = [space for space in psref if spaces[i].is_single_excitation_of (space)]
        if auto_singles:
            ncharge_i = spaces[i].compute_single_excitation_lroots (psref)
        else:
            ncharge_i = ncharge
        lroots[:,i][excfrags] = ncharge_i = np.amin (np.minimum (lroots[:,i][excfrags], ncharge_i))
        lroots[:,i][~excfrags] = 1
        # logging after setup
        spref0 = spaces[psref_ix[0]]
        key = spaces[i].single_excitation_key (spref0)
        keystr = spaces[i].single_excitation_description_string (spref0)
        log.info ("Electron hop %d/%d: %s", i-las1.nroots, len (spaces)-las1.nroots, keystr)
        spaces[i].table_printlog (lroots=lroots[:,i])
        log.debug ("is connected to reference spaces:")
        for space in psref[:nref_pure]:
            space.table_printlog (tverbose=logger.DEBUG)
            log.debug ('by %s', spaces[i].single_excitation_description_string (space))
        if len (psref) > nref_pure:
            log.debug ("as well as spin-excited spaces:")
            for space in psref[nref_pure:]:
                space.table_printlog (tverbose=logger.DEBUG)
                log.debug ('by %s', spaces[i].single_excitation_description_string (space))
        assert (key not in keys), 'Problem enumerating model states! Talk to Matt about it!'
        keys.add (key)
        # throat-clearing into ExcitationPSFCISolver
        ciref = [[] for j in range (nfrags)]
        for k in range (nfrags):
            for space in psref: ciref[k].append (space.ci[k])
        spaces[i].set_entmap_(psref[0])
        psref = [space.get_product_state_solver () for space in psref]
        psexc = ExcitationPSFCISolver (psref, ciref, las2.ncas_sub, las2.nelecas_sub,
                                       stdout=mol.stdout, verbose=mol.verbose,
                                       crash_locmin=crash_locmin, opt=lsi.opt)
        psexc._deactivate_vrv = deactivate_vrv
        norb = spaces[i].nlas
        neleca = spaces[i].neleca
        nelecb = spaces[i].nelecb
        smults = spaces[i].smults
        for k in np.where (excfrags)[0]:
            psexc.set_excited_fragment_(k, (neleca[k],nelecb[k]), smults[k])
        ifrag, afrag, spin = key
        norb_i, norb_a, smult_i, smult_a = norb[ifrag], norb[afrag], smults[ifrag], smults[afrag]
        nelec_i, nelec_a = (neleca[ifrag],nelecb[ifrag]), (neleca[afrag],nelecb[afrag])
        # Going into psexc.kernel, they have to be in lexical order
        ci0 = ci_ch_ias = ci_ch[ifrag][afrag][spin]
        if ci0[0] is not None: 
            ci0[0] = mdown (ci0[0], norb_i, nelec_i, smult_i)
            if lroots[ifrag,i] == 1 and ci0[0].ndim==3: ci0[0] = ci0[0][0]
        if ci0[1] is not None:
            ci0[1] = mdown (ci0[1], norb_a, nelec_a, smult_a)
            if lroots[afrag,i] == 1 and ci0[1].ndim==3: ci0[1] = ci0[1][0]
        ci0 = [ci0[int (afrag<ifrag)], ci0[int (ifrag<afrag)]]
        conv, e_roots[i], ci1, disc_svals_max = psexc.kernel (
            h1, h2, ecore=h0, ci0=ci0, max_cycle_macro=lsi.max_cycle_macro,
            conv_tol_self=lsi.conv_tol_self, nroots=ncharge_i
        )
        ci_ch_ias[0] = mup (ci1[ifrag], norb_i, nelec_i, smult_i)
        if lroots[ifrag,i]==1 and ci_ch_ias[0].ndim == 2:
            ci_ch_ias[0] = ci_ch_ias[0][None,:,:]
        ci_ch_ias[1] = mup (ci1[afrag], norb_a, nelec_a, smult_a)
        if lroots[afrag,i]==1 and ci_ch_ias[1].ndim == 2:
            ci_ch_ias[1] = ci_ch_ias[1][None,:,:]
        if len (psref)>1:
            for k in np.where (~excfrags)[0]: ci1[k] = ci1[k][0]
        spaces[i].ci = ci1
        if not conv: log.warn ("CI vectors for charge-separated rootspace %s not converged",keystr)
        converged = converged and conv
        log.info ('Electron hop {} max disc sval: {}'.format (keystr, disc_svals_max))
        max_max_disc = max (max_max_disc, disc_svals_max)
        t0 = log.timer ("Electron hop {}".format (keystr), *t0)
    return converged, ci_ch, max_max_disc

class SpinFlips (object):
    '''For a single fragment, bundle the ci vectors of various spin-flipped states with their
       corresponding quantum numbers. Instances of this object are stored together in a list
       where position indicates fragment identity.'''
    def __init__(self, ci, norb, nelec, spins, smults):
        self.norb = norb
        self.nelec = nelec
        self.ci = ci
        self.spins = spins
        self.smults = smults
        # Assumes you only assigned the m=s case
        for i in range (len (ci)):
            ci, spin, smult = self.ci[i], self.spins[i], self.smults[i]
            if smult>1:
                neleca = (self.nelec + (smult-1)) // 2
                nelecb = (self.nelec - (smult-1)) // 2
                ci_list = list (ci)
                for ms in range (smult-1):
                    ci_list = [contract_sdown (c, norb, (neleca,nelecb)) for c in ci_list]
                    neleca -= 1
                    nelecb += 1
                    self.ci.append (np.array (ci_list))
                    self.smults.append (smult)
                    self.spins.append (neleca-nelecb)
                
            

def all_spin_flips (lsi, las, ci_sf, nspin=1, ham_2q=None):
    # NOTE: this actually only uses the -first- rootspace in las, so it can be done before
    # the initial spin shuffle
    log = logger.new_logger (lsi, lsi.verbose)
    norb_f = las.ncas_sub
    spaces = list_spaces (las)
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
    if ham_2q is None:
        h0, h1, h2 = lsi.ham_2q ()
    else:
        h0, h1, h2 = ham_2q
    casdm1s = las.make_casdm1s ()
    f1 = h1 + np.tensordot (h2, casdm1s.sum (0), axes=2)
    f1 = f1[None,:,:] - np.tensordot (casdm1s, h2, axes=((1,2),(2,1)))
    i = 0
    auto_singles = isinstance (nspin, str) and 's' in nspin.lower ()
    nup0 = np.minimum (spaces[0].nelecd, spaces[0].nholeu)
    ndn0 = np.minimum (spaces[0].nelecu, spaces[0].nholed)
    converged = True
    if not auto_singles: # integer supplied by caller
        nup0[:] = nspin
        ndn0[:] = nspin
    for ifrag, (norb, nelec, spin, smult) in enumerate (zip (norb0, nelec0, spins0, smults0)):
        j = i + norb
        h2_i = h2[i:j,i:j,i:j,i:j]
        lasdm1s = casdm1s[:,i:j,i:j]
        h1_i = (f1[:,i:j,i:j] - np.tensordot (h2_i, lasdm1s.sum (0))[None,:,:]
                + np.tensordot (lasdm1s, h2_i, axes=((1,2),(2,1))))
        def cisolve (sm, m2, nroots, ci0):
            neleca = (nelec + m2) // 2
            nelecb = (nelec - m2) // 2
            solver = csf_solver (las.mol, smult=sm).set (nelec=(neleca,nelecb), norb=norb)
            solver.check_transformer_cache ()
            nroots = min (nroots, solver.transformer.ncsf)
            ci_list = solver.kernel (h1_i, h2_i, norb, (neleca,nelecb), ci0=ci0, nroots=nroots)[1]
            if nroots==1: ci_list = [ci_list,]
            ci_arr = np.array ([mup (c, norb, (neleca,nelecb), sm) for c in ci_list])
            return np.all (solver.converged), ci_arr
        smults1_i = []
        spins1_i = []
        ci1_i = []
        if smult > 2: # spin-lowered
            log.info ("LASSIS fragment %d spin down (%de,%do;2S+1=%d)",
                      ifrag, nelec, norb, smult-2)
            smults1_i.append (smult-2)
            spins1_i.append (smult-3)
            ci0 = ci_sf[ifrag][0]
            m2 = np.sign (spin) * (abs (spin) - 2) if abs (spin) > 1 else spin
            conv, ci1_i_down = cisolve (smult-2, m2, ndn0[ifrag], ci0)
            if not conv: log.warn ("CI vectors for spin-lowering of fragment %i not converged",
                                   ifrag)
            converged = converged & conv
            ci_sf[ifrag][0] = ci1_i_down
            ci1_i.append (ci1_i_down)
        min_npair = max (0, nelec-norb)
        max_smult = (nelec - 2*min_npair) + 1
        if smult < max_smult: # spin-raised
            log.info ("LASSIS fragment %d spin up (%de,%do;2S+1=%d)",
                      ifrag, nelec, norb, smult+2)
            smults1_i.append (smult+2)
            spins1_i.append (smult+1)
            ci0 = ci_sf[ifrag][1]
            m2 = np.sign (spin) * (abs (spin) + 2)
            conv, ci1_i_up = cisolve (smult+2, m2, nup0[ifrag], ci0)
            if not conv: log.warn ("CI vectors for spin-raising of fragment %i not converged",
                                   ifrag)
            converged = converged & conv
            ci_sf[ifrag][1] = ci1_i_up
            ci1_i.append (ci1_i_up)
        smults1.append (smults1_i)
        spins1.append (spins1_i)
        ci1.append (ci1_i)
        i = j
    spin_flips = [SpinFlips (c,no,ne,m,s) for c,no,ne,m,s in zip (ci1,norb0,nelec0,spins1,smults1)]
    return converged, spin_flips, ci_sf

def _spin_flip_products (spaces, spin_flips, nroots_ref=1, frozen_frags=None):
    # NOTE: this actually only uses the -first- rootspace in las, so it can be done before
    # the initial spin shuffle
    '''Combine spin-flip excitations in all symmetrically permissible ways'''
    if spin_flips is None or len (spin_flips)==0: return spaces
    spaces_ref = spaces[:nroots_ref]
    spins3 = [she.spins for she in spin_flips]
    smults3 = [she.smults for she in spin_flips]
    ci3 = [she.ci for she in spin_flips]
    nelec0 = spaces[0].nelec
    smults0 = spaces[0].smults
    nfrags = spaces[0].nfrag
    spin = spaces[0].spins.sum ()
    if frozen_frags is None: frozen_frags = np.zeros (nfrags, dtype=bool)
    for ifrag in range (nfrags):
        if frozen_frags[ifrag]: continue
        new_spaces = []
        m3, s3, c3 = spins3[ifrag], smults3[ifrag], ci3[ifrag]
        for space in spaces:
            # I want to inject the spin-flip into all distinct references,
            # but if two references differ only in ifrag then this would
            # generate duplicates. The two lines below filter this case.
            if space.nelec[ifrag] != nelec0[ifrag]: continue
            if space.smults[ifrag] != smults0[ifrag]: continue
            for m3i, s3i, c3i in zip (m3, s3, c3):
                new_spaces.append (space.single_fragment_spin_change (
                    ifrag, s3i, m3i, ci=c3i))
        spaces += new_spaces
    # Filter by ms orthogonality
    spaces = [space for space in spaces if space.spins.sum () == spin]
    # Filter by smult orthogonality
    spaces = [space for space in spaces 
              if (not (all (space.is_orthogonal_by_smult (spaces_ref))))]
    seen = set ()
    # Filter duplicates!
    spaces = [space for space in spaces if not ((space in seen) or seen.add (space))]
    return spaces

def _spin_shuffle_ci_(spaces, spin_flips, nroots_ref, nroots_refc):
    '''Memory-efficient version of the function spaces._spin_shuffle_ci_.
    Based on the fact that we know there has only been one independent set
    of vectors per fragment Hilbert space and that all possible individual
    fragment spins must be accounted for already, so we are just recombining
    them.'''
    old_idx = []
    new_idx = []
    nfrag = spaces[0].nfrag
    for ix, space in enumerate (spaces):
        if space.has_ci ():
            old_idx.append (ix)
        else:
            assert (ix >= nroots_refc)
            new_idx.append (ix)
            space.ci = [None for ifrag in range (space.nfrag)]
    # Prepare charge-hop szrots
    spaces_1c = spaces[nroots_ref:nroots_refc]
    spaces_1c = [space for space in spaces_1c if len (space.entmap)==1]
    ci_szrot_1c = []
    for ix, space in enumerate (spaces_1c):
        ifrag, jfrag = space.entmap[0] # must be a tuple of length 2
        ci_szrot_1c.append (space.get_ci_szrot (ifrags=(ifrag,jfrag)))
    charges0 = spaces[0].charges
    smults0 = spaces[0].smults
    # Prepare reference szrots
    ci_szrot_ref = spaces[0].get_ci_szrot ()
    for ix in new_idx:
        idx = spaces[ix].excited_fragments (spaces[0])
        space = spaces[ix]
        for ifrag in np.where (~idx)[0]:
            space.ci[ifrag] = spaces[0].ci[ifrag]
        for ifrag in np.where (idx)[0]:
            if space.charges[ifrag] != charges0[ifrag]: continue
            if space.smults[ifrag] != smults0[ifrag]:
                sf = spin_flips[ifrag]
                iflp = sf.smults == space.smults[ifrag]
                iflp &= sf.spins == space.spins[ifrag]
                assert (np.count_nonzero (iflp) == 1)
                iflp = np.where (iflp)[0][0]
                space.ci[ifrag] = sf.ci[iflp]
            else: # Reference-state spin-shuffles
                space.ci[ifrag] = ci_szrot_ref[ifrag][space.spins[ifrag]]
        for (ci_i, ci_j), sp_1c in zip (ci_szrot_1c, spaces_1c):
            ijfrag = sp_1c.entmap[0]
            if ijfrag not in spaces[ix].entmap: continue
            if np.any (sp_1c.charges[list(ijfrag)] != space.charges[list(ijfrag)]): continue
            if np.any (sp_1c.smults[list(ijfrag)] != space.smults[list(ijfrag)]): continue
            ifrag, jfrag = ijfrag
            assert (space.ci[ifrag] is None)
            assert (space.ci[jfrag] is None)
            space.ci[ifrag] = ci_i[space.spins[ifrag]]
            space.ci[jfrag] = ci_j[space.spins[jfrag]]
        assert (space.has_ci ()), '{} {} {} {}'.format (space.charges, space.smults, space.spins, charges0)
    return spaces

def spin_flip_products (las, spaces, spin_flips, nroots_ref=1):
    '''Inject spin-flips into spaces in all possible ways, carry out a spin shuffle, and log'''
    log = logger.new_logger (las, las.verbose)
    nspaces = len (spaces)
    spaces = _spin_flip_products (spaces, spin_flips, nroots_ref=nroots_ref)
    nfrags = spaces[0].nfrag
    spaces = _spin_shuffle (spaces)
    spaces = _spin_shuffle_ci_(spaces, spin_flips, nroots_ref, nspaces)
    log.debug ("LASSIS spin-excitation spaces: %d-%d", nspaces, len (spaces)-1)
    for i, space in enumerate (spaces[nspaces:]):
        if np.any (space.nelec != spaces[0].nelec):
            log.debug ("Spin/charge-excitation space %d:", i+nspaces)
        else:
            log.debug ("Spin-excitation space %d:", i+nspaces)
        space.table_printlog (tverbose=logger.DEBUG)
    return spaces

def charge_excitation_products (lsi, spaces, nroots_ref=0, space0=None):
    t0 = (logger.process_clock (), logger.perf_counter ())
    log = logger.new_logger (lsi, lsi.verbose)
    mol = lsi.mol
    nfrags = lsi.nfrags
    if space0 is None: space0 = spaces[0]
    i0, j0 = i, j = nroots_ref, len (spaces)
    for product_order in range (2, (nfrags//2)+1):
        seen = set ()
        for i_list in itertools.combinations (range (i,j), product_order):
            p_list = [spaces[ip] for ip in i_list]
            nonorth = False
            for p, q in itertools.combinations (p_list, 2):
                if not orthogonal_excitations (p, q, space0):
                    nonorth = True
                    break
            if nonorth: continue
            p = p_list[0]
            for q in p_list[1:]:
                p = combine_orthogonal_excitations (p, q, space0)
            spaces.append (p)
            log.debug ("Electron hop product space %d (product of %s)", len (spaces) - 1, str (i_list))
            spaces[-1].table_printlog (tverbose=logger.DEBUG)
    assert (len (spaces) == len (set (spaces)))
    log.timer ("LASSIS charge-hop product generation", *t0)
    return spaces

def as_scanner(lsi):
    '''Generating a scanner for LASSIS PES.
    
    The returned solver is a function. This function requires one argument
    "mol" as input and returns total LASSIS energy.

    The solver will automatically use the results of last calculation as the
    initial guess of the new calculation.  All parameters of LASSIS object
    are automatically applied in the solver.
    
    Note scanner has side effects.  It may change many underlying objects
    (_scf, with_df, with_x2c, ...) during calculation.
    ''' 
    if isinstance(lsi, lib.SinglePointScanner):
        return lsi
        
    logger.info(lsi, 'Create scanner for %s', lsi.__class__)
    name = lsi.__class__.__name__ + LASSIS_Scanner.__name_mixin__
    return lib.set_class(LASSIS_Scanner(lsi), (LASSIS_Scanner, lsi.__class__), name)
        
class LASSIS_Scanner(lib.SinglePointScanner):
    def __init__(self, lsi, state=0):
        self.__dict__.update(lsi.__dict__)
        self._las = lsi._las.as_scanner()
        self._scan_state = state

    def __call__(self, mol_or_geom, **kwargs):
        if isinstance(mol_or_geom, gto.MoleBase):
            mol = mol_or_geom
        else:
            mol = self.mol.set_geom_(mol_or_geom, inplace=False)
    
        self.reset (mol)
        for key in ('with_df', 'with_x2c', 'with_solvent', 'with_dftd3'):
            sub_mod = getattr(self, key, None)
            if sub_mod:
                sub_mod.reset(mol)

        las_scanner = self._las
        las_scanner(mol)
        self.mol = mol
        self.mo_coeff = las_scanner.mo_coeff
        e_tot = self.kernel()[0][self._scan_state]
        if hasattr (e_tot, '__len__'):
            e_tot = np.average (e_tot)
        return e_tot

class LASSIS (LASSI):
    def __init__(self, las, ncharge='s', nspin='s', sa_heff=True, deactivate_vrv=False,
                 crash_locmin=False, opt=1, **kwargs):
        '''
        Key attributes:
            _las : instance of class `LASCINoSymm`
                The encapsulated LASSCF wave function. The CI vectors of the reference state are,
                i.e., _las.get_single_state_las (state=0).ci.
            ci_spin_flips : doubly nested list of ndarrays
                Element [i][s] are the spin-flip CI vectors of the ith fragment in the direction
                s = 0,1 = -,+.
            ci_charge_hops: quadruply nested list of ndarrays
                Element [i][a][s][p] are charge-hop CI vectors for an electron hopping from the
                ith to the ath fragment for spin case s = 0,1,2,3 = --,-+,+-,++, and fragment
                p = 0,1 = i,a.
            entmaps: list of length nroots of tuple of tuples
                Tracks which fragments are entangled to one another in each rootspace
        '''
        self.ncharge = ncharge
        self.nspin = nspin
        self.sa_heff = sa_heff
        self.deactivate_vrv = deactivate_vrv
        self.crash_locmin = crash_locmin
        self.e_states_meaningless = True # a tag to silence an invalid warning
        LASSI.__init__(self, las, opt=opt, **kwargs)
        self.max_cycle_macro = 50
        self.conv_tol_self = 1e-8
        self.ci_spin_flips = [[None for s in range (2)] for i in range (self.nfrags)]
        self.ci_charge_hops = [[[[None,None] for s in range (4)]
                                for a in range (self.nfrags)]
                               for i in range (self.nfrags)]
        self._cached_ham_2q = None
        self.ci = None
        if las.nroots>1:
            logger.warn (self, ("Only the first LASSCF state is used by LASSIS! "
                                "Other states are discarded!"))

    def ham_2q (self, *args, **kwargs):
        if self._cached_ham_2q is not None: return self._cached_ham_2q
        return super().ham_2q (*args, **kwargs)

    def kernel (self, ncharge=None, nspin=None, sa_heff=None, deactivate_vrv=None,
                crash_locmin=None, **kwargs):
        t0 = (logger.process_clock (), logger.perf_counter ())
        log = logger.new_logger (self, self.verbose)
        h0, h1, h2 = self.ham_2q ()
        t1 = log.timer ("LASSIS integral transformation", *t0)
        with lib.temporary_env (self, _cached_ham_2q=(h0,h1,h2)):
            self.converged = self.prepare_states_(ncharge=ncharge, nspin=nspin,
                                                  sa_heff=sa_heff, deactivate_vrv=deactivate_vrv,
                                                  crash_locmin=crash_locmin)
            t1 = log.timer ("LASSIS state preparation", *t1)
            self.e_roots, self.si = self.eig (**kwargs)
            t1 = log.timer ("LASSIS diagonalization", *t1)
        log.timer ("LASSIS", *t0)
        return self.e_roots, self.si

    def get_ci_ref (self):
        las = self._las.get_single_state_las (state=0)
        space0 = list_spaces (las)[0]
        ci_ref = []
        for i in range (self.nfrags):
            ci_ref.append (mup (space0.ci[i], space0.nlas[i], (space0.neleca[i], space0.nelecb[i]),
                                space0.smults[i]))
        return ci_ref

    def get_las_of_ci_ref (self, ci_ref):
        las = self._las.get_single_state_las (state=0)
        space0 = list_spaces (las)[0]
        for i in range (las.nfrags):
            nelec, smult = (space0.neleca[i], space0.nelecb[i]), space0.smults[i]
            las.ci[i][0] = space0.ci[i] = mdown (ci_ref[i], las.ncas_sub[i], nelec, smult)
        return las

    def prepare_states_(self, ncharge=None, nspin=None, sa_heff=None, deactivate_vrv=None,
                        crash_locmin=None, **kwargs):
        if ncharge is None: ncharge = self.ncharge
        if nspin is None: nspin = self.nspin
        if sa_heff is None: sa_heff = self.sa_heff
        if deactivate_vrv is None: deactivate_vrv = self.deactivate_vrv
        if crash_locmin is None: crash_locmin = self.crash_locmin
        log = logger.new_logger (self, self.verbose)

        ci_ref = self.get_ci_ref ()
        ci_sf = self.ci_spin_flips
        ci_ch = self.ci_charge_hops
        self.converged, ci_sf, ci_ch, self.max_disc_sval = self.prepare_fbf (
            ci_ref, ci_sf, ci_ch, ncharge=ncharge, nspin=nspin, sa_heff=sa_heff,
            deactivate_vrv=deactivate_vrv, crash_locmin=crash_locmin
        )
        self.ci_spin_flips = ci_sf
        self.ci_charge_hops = ci_ch

        las, self.entmaps = self.prepare_model_states (ci_ref, ci_sf, ci_ch)
        #self.__dict__.update(las.__dict__) # Unsafe
        self.fciboxes = las.fciboxes
        self.ci = las.ci
        self.nroots = las.nroots
        self.weights = las.weights
        self.e_lexc = las.e_lexc
        self.e_states = las.e_states
        log.info ('LASSIS model state summary: %d rootspaces; %d model states; converged? %s',
                  self.nroots, self.get_lroots ().prod (0).sum (), str (self.converged))
        log.info ('LASSIS overall max disc sval: %e', self.max_disc_sval)
        return self.converged

    def energy_tot (self, mo_coeff=None, ci_ref=None, ci_sf=None, ci_ch=None, si=None, soc=None):
        if ci_ref is None: ci_ref = self.get_ci_ref ()
        if ci_sf is None: ci_sf = self.ci_spin_flips
        if ci_ch is None: ci_ch = self.ci_charge_hops
        if soc is None: soc = self.soc
        las = self.prepare_model_states (ci_ref, ci_sf, ci_ch)[0]
        ci = las.ci
        self.fciboxes = las.fciboxes # TODO: set this at initialization
        return LASSI.energy_tot (self, mo_coeff=mo_coeff, ci=ci, si=si, soc=soc)

    def get_lroots (self, ci=None):
        if ci is None: ci = self.ci
        if ci is None:
            with lib.temporary_env (self, max_cycle_macro=0):
                self.prepare_states_()
            ci = self.ci
        assert (ci is not None)
        return LASSI.get_lroots (self, ci=ci)

    def get_raw2orth (self, ci_ref=None, ci_sf=None, ci_ch=None, soc=None, opt=None):
        if ci_ref is None: ci_ref = self.get_ci_ref ()
        if ci_sf is None: ci_sf = self.ci_spin_flips
        if ci_ch is None: ci_ch = self.ci_charge_hops
        if soc is None: soc = self.soc
        las = self.prepare_model_states (ci_ref, ci_sf, ci_ch)[0]
        ci = las.ci
        self.fciboxes = las.fciboxes # TODO: set this at initialization
        return LASSI.get_raw2orth (self, ci=ci, soc=soc, opt=opt)

    eig = LASSI.kernel
    as_scanner = as_scanner
    prepare_fbf = prepare_fbf
    prepare_model_states = prepare_model_states

    def get_ref_fbf_rootspaces (self, ifrag):
        '''Identify which rootspaces correspond to the reference wave function for a given
        fragment.

        Args:
            ifrag : integer

        Returns:
            idx : ndarray of integer
                Indices of the corresponding rootspaces
            nelec_rs : ndarray of shape (len (idx), 2)
                neleca,nelecb in the corresponding rootspaces for the purpose of mdowning
        '''
        ref_space = list_spaces (self._las.get_single_state_las (state=0))[0]
        nelec = ref_space.nelec[ifrag]
        smult = ref_space.smults[ifrag]
        nelec_rs = self.get_nelec_frs ()[ifrag]
        smult_r = self.get_smult_fr ()[ifrag]
        idx = (nelec_rs.sum (1) == nelec) & (smult_r == smult)
        idx = np.where (idx)[0]
        return idx, nelec_rs[idx,:]

    def get_sf_fbf_rootspaces (self, ifrag, spin):
        '''Identify which rootspaces correspond to a spin-flip up or down for a given fragment.

        Args:
            ifrag : integer
            spin : integer
                0,1 -> -,+

        Returns:
            idx : ndarray of integer
                Indices of the corresponding rootspaces
            nelec_rs : ndarray of shape (len (idx), 2)
                neleca,nelecb in the corresponding rootspaces for the purpose of mdowning
        '''
        ref_space = list_spaces (self._las.get_single_state_las (state=0))[0]
        nelec = ref_space.nelec[ifrag]
        smult = ref_space.smults[ifrag] - 2 + 4*spin
        nelec_rs = self.get_nelec_frs ()[ifrag]
        smult_r = self.get_smult_fr ()[ifrag]
        idx = (nelec_rs.sum (1) == nelec) & (smult_r == smult)
        idx = np.where (idx)[0]
        return idx, nelec_rs[idx,:]
        
    def get_ch_fbf_rootspaces (self, ifrag, afrag, spin):
        '''Identify which rootspaces correspond to a given charge hop.

        Args:
            ifrag : integer
                Source fragment
            afrag : integer
                Destination fragment
            spin : integer
                0,1,2,3 -> --,-+,+-,++

        Returns:
            idx : ndarray of integer
                Indices of the corresponding rootspaces
            nelec_i_rs : ndarray of shape (len (idx), 2)
                neleca,nelecb of the source fragment in the corresponding rootspaces for the
                purpose of mdowning
            nelec_a_rs : ndarray of shape (len (idx), 2)
                neleca,nelecb of the dest fragment in the corresponding rootspaces for the
                purpose of mdowning
        '''
        ref_space = list_spaces (self._las.get_single_state_las (state=0))[0]
        nelec_i = ref_space.nelec[ifrag] - 1
        nelec_a = ref_space.nelec[afrag] + 1
        smult_i = ref_space.smults[ifrag] - 1 + 2*(spin//2)
        smult_a = ref_space.smults[afrag] - 1 + 2*(spin%2)
        nelec_frs = self.get_nelec_frs ()
        smult_fr = self.get_smult_fr ()
        idx  = (nelec_frs[ifrag].sum (1) == nelec_i) & (smult_fr[ifrag] == smult_i)
        idx &= (nelec_frs[afrag].sum (1) == nelec_a) & (smult_fr[afrag] == smult_a)
        idx = np.where (idx)[0]
        idx2 = np.asarray ([tuple(set((ifrag,afrag))) in self.entmaps[i] for i in idx], dtype=bool)
        idx = idx[idx2]
        return idx, nelec_frs[ifrag][idx,:], nelec_frs[afrag][idx,:]

    def get_fbf_idx (self, ifrag, ci_sf=None, ci_ch=None):
        if ci_sf is None: ci_sf = self.ci_spin_flips
        if ci_ch is None: ci_ch = self.ci_charge_hops
    
        idx_ref = np.array ([0,1])
        i = 1
        ci_sf = ci_sf[ifrag]
        idx_sf = -np.ones ((2,2), dtype=int)
        for s, ci in enumerate (ci_sf):
            j = i
            if ci is not None:
                j = i + len (ci)
            idx_sf[s,:] = [i,j]
            i = j

        idx_ch = -np.ones ((2,self.nfrags,4,2), dtype=int)
        for afrag, ci_ch_a in enumerate (ci_ch[ifrag]):
            for s, ci in enumerate (ci_ch_a):
                j = i
                if ci[0] is not None:
                    j = i + len (ci[0])
                idx_ch[0,afrag,s,:] = [i,j]
                i = j
        afrag = ifrag
        for jfrag, ci_ch_j in enumerate (ci_ch):
            for s, ci in enumerate (ci_ch_j[afrag]):
                j = i
                if ci[1] is not None:
                    j = i + len (ci[1])
                idx_ch[1,jfrag,s,:] = [i,j]
                i = j

        return idx_ref, idx_sf, idx_ch

    def get_fbf_ovlp (self, ifrag, ci_ref=None, ci_sf=None, ci_ch=None):
        if ci_ref is None: ci_ref = self.get_ci_ref ()
        if ci_sf is None: ci_sf = self.ci_spin_flips
        if ci_ch is None: ci_ch = self.ci_charge_hops
        idx_ref, idx_sf, idx_ch = self.get_fbf_idx (ifrag, ci_sf=ci_sf, ci_ch=ci_ch)
        nstates = idx_ch[-1,-1,-1,-1]
        ovlp = np.zeros ((nstates, nstates), dtype=ci_ref[ifrag].dtype)
        i, j = idx_ref
        ovlp[i:j,i:j] = np.dot (ci_ref[ifrag].conj ().flat, ci_ref[ifrag].flat)
        for (i,j), ci in zip (idx_sf, ci_sf[ifrag]):
            if j==i: continue
            c = ci.reshape (j-i,-1)
            ovlp[i:j,i:j] = np.dot (c.conj (), c.T)
        for afrag, idx_ch_a in enumerate (idx_ch[0]):
            for s1, (i,j) in enumerate (idx_ch_a):
                if j==i: continue
                c1 = ci_ch[ifrag][afrag][s1][0].reshape (j-i,-1)
                ovlp[i:j,i:j] = np.dot (c1.conj (), c1.T)
                for bfrag, idx_ch_b in enumerate (idx_ch[0]):
                    for s2, (k,l) in enumerate (idx_ch_b):
                        if k==l: continue
                        if ((s1//2)!=(s2//2)): continue
                        c2 = ci_ch[ifrag][bfrag][s2][0].reshape (l-k,-1)
                        ovlp[i:j,k:l] = np.dot (c1.conj (), c2.T)
        afrag = ifrag
        for ifrag, idx_ch_i in enumerate (idx_ch[1]):
            for s1, (i,j) in enumerate (idx_ch_i):
                if j==i: continue
                c1 = ci_ch[ifrag][afrag][s1][1].reshape (j-i,-1)
                ovlp[i:j,i:j] = np.dot (c1.conj (), c1.T)
                for jfrag, idx_ch_j in enumerate (idx_ch[1]):
                    for s2, (k,l) in enumerate (idx_ch_j):
                        if k==l: continue
                        if ((s1%2)!=(s2%2)): continue
                        c2 = ci_ch[jfrag][afrag][s2][1].reshape (l-k,-1)
                        ovlp[i:j,k:l] = np.dot (c1.conj (), c2.T)
        return ovlp
            
    def make_fbfdm1 (self, pfrag, si=None, state=0):
        from mrh.my_pyscf.lassi.sitools import decompose_sivec_by_rootspace, _make_sdm1, _trans_sdm1
        if si is None: si = self.si
        states = np.atleast_1d (state)
        nstates = len (states)
        space_weights, state_coeffs = decompose_sivec_by_rootspace (self, si[:,states])[:2]

        lroots = self.get_lroots ()
        nelec_rfs = self.get_nelec_frs ().transpose (1,0,2)
        smult_rf = self.get_smult_fr ().T
        def my_make_sdm1 (ix,jx):
            if np.any (nelec_rfs[ix]!=nelec_rfs[jx]): return 0
            if np.any (smult_rf[ix]!=smult_rf[jx]): return 0
            ovlp = []
            for f in range (self.nfrags):
                ci_i = self.ci[f][ix].reshape (lroots[f,ix],-1)
                ci_j = self.ci[f][jx].reshape (lroots[f,jx],-1)
                ovlp.append (ci_i.conj () @ ci_j.T)
            wgt = np.sqrt (space_weights[ix]*space_weights[jx]) 
            ddm = _trans_sdm1 (state_coeffs[ix], lroots[:,ix], 
                               state_coeffs[jx], lroots[:,jx],
                               ovlp, pfrag)
            return np.tensordot (wgt, ddm, axes=1) / nstates


        fbf_idx = self.get_fbf_idx (pfrag)
        nbas = fbf_idx[-1][-1,-1,-1,-1]
        fbfdm1 = np.zeros ((nbas,nbas), dtype=si.dtype)

        idx = self.get_ref_fbf_rootspaces (pfrag)[0]
        i, j = fbf_idx[0]
        for ix, jx in itertools.product (idx, repeat=2):
            fbfdm1[i:j,i:j] += my_make_sdm1 (ix,jx)

        for s in range (2):
            if self.ci_spin_flips[pfrag][s] is None: continue
            idx = self.get_sf_fbf_rootspaces (pfrag,s)[0]
            i, j = fbf_idx[1][s]
            for ix, jx in itertools.product (idx, repeat=2):
                fbfdm1[i:j,i:j] += my_make_sdm1 (ix, jx)

        ifrag = pfrag
        for s in range (4):
            for afrag in range (self.nfrags):
                i_present = self.ci_charge_hops[ifrag][afrag][s][0] is not None
                a_present = self.ci_charge_hops[ifrag][afrag][s][1] is not None
                if not (i_present and a_present): continue
                idx = self.get_ch_fbf_rootspaces (ifrag,afrag,s)[0]
                i, j = fbf_idx[2][0,afrag,s]
                for ix, jx in itertools.product (idx, repeat=2):
                    fbfdm1[i:j,i:j] += my_make_sdm1 (ix, jx)
                for bfrag in range (self.nfrags):
                    if bfrag==afrag: continue
                    b_present = self.ci_charge_hops[ifrag][bfrag][s][1] is not None
                    if not b_present: continue
                    jdx = self.get_ch_fbf_rootspaces (ifrag,bfrag,s)[0]
                    k, l = fbf_idx[2][0,bfrag,s]
                    for ix, jx in itertools.product (idx, jdx):
                        fbfdm1[i:j,k:l] += my_make_sdm1 (ix, jx)
        afrag = pfrag
        for s in range (4):
            for ifrag in range (self.nfrags):
                i_present = self.ci_charge_hops[ifrag][afrag][s][0] is not None
                a_present = self.ci_charge_hops[ifrag][afrag][s][1] is not None
                if not (i_present and a_present): continue
                idx = self.get_ch_fbf_rootspaces (ifrag,afrag,s)[0]
                i, j = fbf_idx[2][1,ifrag,s]
                for ix, jx in itertools.product (idx, repeat=2):
                    fbfdm1[i:j,i:j] += my_make_sdm1 (ix, jx)
                for jfrag in range (self.nfrags):
                    if jfrag==ifrag: continue
                    j_present = self.ci_charge_hops[jfrag][afrag][s][1] is not None
                    if not j_present: continue
                    jdx = self.get_ch_fbf_rootspaces (jfrag,afrag,s)[0]
                    k, l = fbf_idx[2][1,jfrag,s]
                    for ix, jx in itertools.product (idx, jdx):
                        fbfdm1[i:j,k:l] += my_make_sdm1 (ix, jx)

        return fbfdm1


