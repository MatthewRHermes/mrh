import numpy as np
from pyscf.fci.direct_spin1 import _unpack_nelec
from pyscf.fci import cistring
from pyscf.lib import logger
from pyscf.lo.orth import vec_lowdin
from pyscf import symm
from mrh.my_pyscf.fci.spin_op import contract_sdown, contract_sup
from mrh.my_pyscf.fci.csfstring import CSFTransformer
from mrh.my_pyscf.fci.csfstring import ImpossibleSpinError
import itertools

class SingleLASRootspace (object):
    def __init__(self, las, spins, smults, charges, weight, nlas=None, nelelas=None, stdout=None,
                 verbose=None, ci=None):
        if nlas is None: nlas = las.ncas_sub
        if nelelas is None: nelelas = [sum (_unpack_nelec (x)) for x in las.nelecas_sub]
        if stdout is None: stdout = las.stdout
        if verbose is None: verbose = las.verbose
        self.las = las
        self.nlas, self.nelelas = np.asarray (nlas), np.asarray (nelelas)
        self.nfrag = len (nlas)
        self.spins, self.smults = np.asarray (spins), np.asarray (smults)
        self.charges = np.asarray (charges)
        self.weight = weight
        self.stdout, self.verbose = stdout, verbose
        self.ci = ci

        self.nelec = self.nelelas - self.charges
        self.neleca = (self.nelec + self.spins) // 2
        self.nelecb = (self.nelec - self.spins) // 2
        self.nhole = 2*self.nlas - self.nelec 
        self.nholea = self.nlas - self.neleca
        self.nholeb = self.nlas - self.nelecb

    def __eq__(self, other):
        if self.nfrag != other.nfrag: return False
        return (np.all (self.spins==other.spins) and 
                np.all (self.smults==other.smults) and
                np.all (self.charges==other.charges))

    def __hash__(self):
        return hash (tuple ([self.nfrag,] + list (self.spins) + list (self.smults)
                            + list (self.charges)))

    def possible_excitation (self, i, a, s):
        i, a, s = np.atleast_1d (i, a, s)
        idx_a = (s == 0)
        ia, nia = np.unique (i[idx_a], return_counts=True)
        if np.any (self.neleca[ia] < nia): return False
        aa, naa = np.unique (a[idx_a], return_counts=True)
        if np.any (self.nholea[aa] < naa): return False
        idx_b = (s == 1)
        ib, nib = np.unique (i[idx_b], return_counts=True)
        if np.any (self.nelecb[ib] < nib): return False
        ab, nab = np.unique (a[idx_b], return_counts=True)
        if np.any (self.nholeb[ab] < nab): return False
        return True

    def get_single (self, i, a, m, si, sa):
        charges = self.charges.copy ()
        spins = self.spins.copy ()
        smults = self.smults.copy ()
        charges[i] += 1
        charges[a] -= 1
        dm = 1 - 2*m
        spins[i] -= dm
        spins[a] += dm
        smults[i] += si
        smults[a] += sa
        log = logger.new_logger (self, self.verbose)
        i_neleca = (self.nelelas[i]-charges[i]+spins[i]) // 2
        i_nelecb = (self.nelelas[i]-charges[i]-spins[i]) // 2
        a_neleca = (self.nelelas[a]-charges[a]+spins[a]) // 2
        a_nelecb = (self.nelelas[a]-charges[a]-spins[a]) // 2
        i_ncsf = CSFTransformer (self.nlas[i], i_neleca, i_nelecb, smults[i]).ncsf
        a_ncsf = CSFTransformer (self.nlas[a], a_neleca, a_nelecb, smults[a]).ncsf
        if (a_neleca==self.nlas[a]) and (a_nelecb==self.nlas[a]) and (smults[a]>1):
            raise ImpossibleSpinError ("too few orbitals?", norb=self.nlas[a],
                                       neleca=a_neleca, nelecb=a_nelecb, smult=smults[a])
        if (i_neleca==0) and (i_nelecb==0) and (smults[i]>1):
            raise ImpossibleSpinError ("too few electrons?", norb=self.nlas[i],
                                       neleca=i_neleca, nelecb=i_nelecb, smult=smults[i])
        log.debug ("spin={} electron from {} to {}".format (dm, i, a))
        log.debug ("c,m,s=[{},{},{}]->c,m,s=[{},{},{}]; {},{} CSFs".format (
            self.charges, self.spins, self.smults,
            charges, spins, smults,
            i_ncsf, a_ncsf))
        assert (i_neleca>=0)
        assert (i_nelecb>=0)
        assert (a_neleca>=0)
        assert (a_nelecb>=0)
        assert (i_ncsf)
        assert (a_ncsf)
        return SingleLASRootspace (self.las, spins, smults, charges, 0, nlas=self.nlas,
                               nelelas=self.nelelas, stdout=self.stdout, verbose=self.verbose)

    def get_valid_smult_change (self, i, dneleca, dnelecb):
        assert ((abs (dneleca) + abs (dnelecb)) == 1), 'Function only implemented for +-1 e-'
        dsmult = []
        neleca = self.neleca[i] + dneleca
        nelecb = self.nelecb[i] + dnelecb
        new_2ms = neleca - nelecb
        min_smult = abs (new_2ms)+1
        min_npair = max (0, neleca+nelecb - self.nlas[i])
        max_smult = 1+neleca+nelecb-(2*min_npair)
        if self.smults[i]>min_smult: dsmult.append (-1)
        if self.smults[i]<max_smult: dsmult.append (+1)
        return dsmult

    def get_singles (self):
        log = logger.new_logger (self, self.verbose)
        # move 1 alpha electron
        has_ea = np.where (self.neleca > 0)[0]
        has_ha = np.where (self.nholea > 0)[0]
        singles = []
        for i, a in itertools.product (has_ea, has_ha):
            if i==a: continue
            si_range = self.get_valid_smult_change (i, -1, 0)
            sa_range = self.get_valid_smult_change (a,  1, 0)
            for si, sa in itertools.product (si_range, sa_range):
                try:
                    singles.append (self.get_single (i,a,0,si,sa))
                except ImpossibleSpinError as e:
                    log.debug ('Caught ImpossibleSpinError: {}'.format (e.__dict__))
        # move 1 beta electron
        has_eb = np.where (self.nelecb > 0)[0]
        has_hb = np.where (self.nholeb > 0)[0]
        for i, a in itertools.product (has_eb, has_hb):
            if i==a: continue
            si_range = self.get_valid_smult_change (i, 0, -1)
            sa_range = self.get_valid_smult_change (a, 0,  1)
            for si, sa in itertools.product (si_range, sa_range):
                try:
                    singles.append (self.get_single (i,a,1,si,sa))
                except ImpossibleSpinError as e:
                    log.debug ('Caught ImpossibleSpinError: {}'.format (e.__dict__))
        return singles

    def gen_spin_shuffles (self):
        assert ((np.sum (self.smults - 1) - np.sum (self.spins)) % 2 == 0)
        nflips = (np.sum (self.smults - 1) - np.sum (self.spins)) // 2
        spins_table = (self.smults-1).copy ()[None,:]
        subtrahend = 2*np.eye (self.nfrag, dtype=spins_table.dtype)[None,:,:]
        for i in range (nflips):
            spins_table = spins_table[:,None,:] - subtrahend
            spins_table = spins_table.reshape (-1, self.nfrag)
            # minimum valid value in column i is 1-self.smults[i]
            idx_valid = np.all (spins_table>-self.smults[None,:], axis=1)
            spins_table = spins_table[idx_valid,:]
        for spins in spins_table:
            yield SingleLASRootspace (self.las, spins, self.smults, self.charges, 0, nlas=self.nlas,
                                  nelelas=self.nelelas, stdout=self.stdout, verbose=self.verbose)

    def has_ci (self):
        if self.ci is None: return False
        return all ([c is not None for c in self.ci])

    def get_ci_szrot (self):
        '''Generate the sets of CI vectors in which each vector for each fragment
        has the sz axis rotated in all possible ways.

        Returns:
            ci_sz: list of dict of type {integer: ndarray}
                dict keys are integerified "spin" quantum numbers; i.e., neleca-nelecb.
                dict vals are the corresponding CI vectors
        '''
        ci_sz = []
        for ifrag in range (self.nfrag):
            norb, sz, ci = self.nlas[ifrag], self.spins[ifrag], self.ci[ifrag]
            nelec = self.neleca[ifrag], self.nelecb[ifrag]
            smult = self.smults[ifrag]
            ci_sz_ = {sz: ci}
            ci1 = ci
            nelec1 = nelec
            for sz1 in range (sz-2, -(1+smult), -2):
                ci1 = contract_sdown (ci1, norb, nelec1)
                nelec1 = nelec1[0]-1, nelec1[1]+1
                ci_sz_[sz1] = ci1
            ci1 = ci
            nelec1 = nelec
            for sz1 in range (sz+2, (1+smult), 2):
                ci1 = contract_sup (ci1, norb, nelec1)
                nelec1 = nelec1[0]+1, nelec1[1]-1
                ci_sz_[sz1] = ci1
            ci_sz.append (ci_sz_)
        return ci_sz

    def get_ndet (self):
        return [(cistring.num_strings (self.nlas[i], self.neleca[i]),
                 cistring.num_strings (self.nlas[i], self.nelecb[i]))
                for i in range (self.nfrag)]

    def is_single_excitation_of (self, other):
        # Same charge sector
        if self.nelec.sum () != other.nelec.sum (): return False
        # Same spinpol sector
        if self.spins.sum () != other.spins.sum (): return False
        # Only 2 fragments involved
        idx_exc = self.excited_fragments (other)
        if np.count_nonzero (idx_exc) != 2: return False
        # Only 1 electron hops
        dnelec = self.nelec[idx_exc] - other.nelec[idx_exc]
        if tuple (np.sort (dnelec)) != (-1,1): return False
        # No additional spin fluctuation between the two excited fragments
        dspins = self.spins[idx_exc] - other.spins[idx_exc]
        if tuple (np.sort (dspins)) != (-1,1): return False
        dsmults = np.abs (self.smults[idx_exc] - other.smults[idx_exc])
        if np.any (dsmults != 1): return False
        return True

    def is_spin_shuffle_of (self, other):
        if np.any (self.nelec != other.nelec): return False
        if np.any (self.smults != other.smults): return False
        return self.spins.sum () == other.spins.sum ()

    def excited_fragments (self, other):
        dneleca = self.neleca - other.neleca
        dnelecb = self.nelecb - other.nelecb
        dsmults = self.smults - other.smults
        idx_same = (dneleca==0) & (dnelecb==0) & (dsmults==0)
        return ~idx_same

    def table_printlog (self):
        log = logger.new_logger (self, self.verbose)
        fmt_str = " {:4s}  {:>11s}  {:>4s}  {:>3s}"
        header = fmt_str.format ("Frag", "Nelec,Norb", "2S+1", "Ir")
        log.info (header)
        fmt_str = " {:4d}  {:>11s}  {:>4d}  {:>3s}"
        for ifrag in range (self.nfrag):
            na, nb = self.neleca[ifrag], self.nelecb[ifrag]
            sm, no = self.smults[ifrag], self.nlas[ifrag]
            irid = 0 # TODO: symmetry
            nelec_norb = '{}a+{}b,{}o'.format (na,nb,no)
            irname = symm.irrep_id2name (self.las.mol.groupname, irid)
            log.info (fmt_str.format (ifrag, nelec_norb, sm, irname))

def all_single_excitations (las, verbose=None):
    '''Add states characterized by one electron hopping from one fragment to another fragment
    in all possible ways. Uses all states already present as reference states, so that calling
    this function a second time generates two-electron excitations, etc. The input object is
    not altered in-place. For orbital optimization, all new states have weight = 0; all weights
    of existing states are unchanged.'''
    from mrh.my_pyscf.mcscf.lasci import get_space_info
    from mrh.my_pyscf.mcscf.lasci import LASCISymm
    if verbose is None: verbose=las.verbose
    log = logger.new_logger (las, verbose)
    if isinstance (las, LASCISymm):
        raise NotImplementedError ("Point-group symmetry for LASSI state generator")
    ref_states = [SingleLASRootspace (las, m, s, c, 0) for c,m,s,w in zip (*get_space_info (las))]
    for weight, state in zip (las.weights, ref_states): state.weight = weight
    new_states = []
    for ref_state in ref_states:
        new_states.extend (ref_state.get_singles ())
    seen = set (ref_states)
    all_states = ref_states + [state for state in new_states if not ((state in seen) or seen.add (state))]
    weights = [state.weight for state in all_states]
    charges = [state.charges for state in all_states]
    spins = [state.spins for state in all_states]
    smults = [state.smults for state in all_states]
    #wfnsyms = [state.wfnsyms for state in all_states]
    log.info ('Built {} singly-excited LAS states from {} reference LAS states'.format (
        len (all_states) - len (ref_states), len (ref_states)))
    if len (all_states) == len (ref_states):
        log.warn (("%d reference LAS states exhaust current active space specifications; "
                   "no singly-excited states could be constructed"), len (ref_states))
    return las.state_average (weights=weights, charges=charges, spins=spins, smults=smults)

def spin_shuffle (las, verbose=None):
    '''Add states characterized by varying local Sz in all possible ways without changing
    local neleca+nelecb, local S**2, or global Sz (== sum local Sz) for each reference state.
    After calling this function, assuming no spin-orbit coupling is included, all LASSI
    results should have good global <S**2>, unless there is severe rounding error due to
    degeneracy between states of different S**2. Unlike all_single_excitations, there
    should never be any reason to call this function more than once. For orbital optimization,
    all new states have weight == 0; all weights of existing states are unchanged.'''
    from mrh.my_pyscf.mcscf.lasci import get_space_info
    from mrh.my_pyscf.mcscf.lasci import LASCISymm
    if verbose is None: verbose=las.verbose
    log = logger.new_logger (las, verbose)
    if isinstance (las, LASCISymm):
        raise NotImplementedError ("Point-group symmetry for LASSI state generator")
    ref_states = [SingleLASRootspace (las, m, s, c, 0) for c,m,s,w in zip (*get_space_info (las))]
    for weight, state in zip (las.weights, ref_states): state.weight = weight
    seen = set (ref_states)
    all_states = [state for state in ref_states]
    for ref_state in ref_states:
        for new_state in ref_state.gen_spin_shuffles ():
            if not new_state in seen:
                all_states.append (new_state)
                seen.add (new_state)
    weights = [state.weight for state in all_states]
    charges = [state.charges for state in all_states]
    spins = [state.spins for state in all_states]
    smults = [state.smults for state in all_states]
    #wfnsyms = [state.wfnsyms for state in all_states]
    log.info ('Built {} spin(local Sz)-shuffled LAS states from {} reference LAS states'.format (
        len (all_states) - len (ref_states), len (ref_states)))
    if len (all_states) == len (ref_states):
        log.warn ("no spin-shuffling options found for given LAS states")
    return las.state_average (weights=weights, charges=charges, spins=spins, smults=smults)

def spin_shuffle_ci (las, ci):
    '''Fill out the CI vectors for rootspaces constructed by the spin_shuffle function.
    Unallocated CI vectors (None elements in ci) for rootspaces which have the same
    charge and spin-magnitude strings of rootspaces that do have allocated CI
    vectors are set to the appropriate rotations of the latter. In the event that
    more than one reference state for an unallocated rootspace is identified, the
    rotated vectors are combined and orthogonalized. Unlike running las.lasci (),
    doing this should ALWAYS guarantee good spin quantum number.'''
    from mrh.my_pyscf.mcscf.lasci import get_space_info
    spaces = [SingleLASRootspace (las, m, s, c, 0, ci=[c[ix] for c in ci])
              for ix, (c, m, s, w) in enumerate (zip (*get_space_info (las)))]
    old_ci_sz = []
    old_idx = []
    new_idx = []
    nfrag = las.nfrags
    for ix, space in enumerate (spaces):
        if space.has_ci ():
            old_idx.append (ix)
            old_ci_sz.append (space.get_ci_szrot ())
        else:
            new_idx.append (ix)
    def is_spin_shuffle_ref (sp1, sp2):
        return (np.all (sp1.charges==sp2.charges) and
                np.all (sp1.smults==sp2.smults))
    for ix in new_idx:
        ndet = spaces[ix].get_ndet ()
        ci_ix = [np.zeros ((0,ndet[i][0],ndet[i][1]))
                 for i in range (nfrag)]
        for ci_sz, jx in zip (old_ci_sz, old_idx):
            if not is_spin_shuffle_ref (spaces[ix], spaces[jx]): continue
            for ifrag in range (nfrag):
                c = ci_sz[ifrag][spaces[ix].spins[ifrag]]
                if c.ndim < 3: c = c[None,:,:]
                ci_ix[ifrag] = np.append (ci_ix[ifrag], c, axis=0)
        for ifrag in range (nfrag):
            if ci_ix[ifrag].size==0:
                ci[ifrag][ix] = None
                continue
            lroots, ndeti = ci_ix[ifrag].shape[0], ndet[ifrag]
            if lroots > 1:
                c = vec_lowdin (ci_ix[ifrag].reshape (lroots, ndeti[0]*ndeti[1]))
                ci_ix[ifrag] = c.reshape (lroots, ndeti[0], ndeti[1])
            ci[ifrag][ix] = ci_ix[ifrag]
    return ci

def count_excitations (las0):
    log = logger.new_logger (las0, las0.verbose)
    t = (logger.process_clock(), logger.perf_counter ())
    log.info ("Counting possible LASSI excitation ranks...")
    nroots0 = las0.nroots
    las1 = all_single_excitations (las0, verbose=0)
    nroots1 = las1.nroots
    for ncalls in range (500):
        if nroots1==nroots0: break
        las1 = all_single_excitations (las1, verbose=0)
        nroots0, nroots1 = nroots1, las1.nroots
    if nroots1>nroots0:
        raise RuntimeError ("Max ncalls reached")
    log.info ("Maximum of %d LAS states reached by excitations of rank %d", nroots0, ncalls)
    log.timer ("LAS excitation counting", *t)
    return nroots0, ncalls

def filter_spaces (las, max_charges=None, min_charges=None, max_smults=None, min_smults=None,
                   target_any_smult=None, target_all_smult=None):
    '''Remove rootspaces from a LASSCF method instance that do not satisfy supplied constraints

    Args:
        las : instance of :class:`LASCINoSymm`

    Kwargs:
        max_charges: integer or ndarray of shape (nfrags,)
            Rootspaces with local charges greater than this are removed
        min_charges: integer or ndarray of shape (nfrags,)
            Rootspaces with local charges less than this are removed
        max_smults: integer or ndarray of shape (nfrags,)
            Rootspaces with local spin magnitudes greater than this are removed
        min_smults: integer or ndarray of shape (nfrags,)
            Rootspaces with local spin magnitudes less than this are removed
        target_any_smult: integer or list
            Rootspaces that cannot couple to total spin magnitudes equaling at
            least one of the supplied integers are removed
        target_all_smult: integer or list
            Rootspaces that cannot couple to total spin magnitudes equaling all
            of the supplied integers are removed

    Returns:
        las : instance of :class:`LASCINoSymm`
            A copy is created
    '''
    log = logger.new_logger (las, las.verbose)
    from mrh.my_pyscf.mcscf.lasci import get_space_info
    charges, spins, smults, wfnsyms = get_space_info (las)
    idx = np.ones (las.nroots, dtype=bool)
    if target_any_smult is not None:
        target_any_s2 = np.asarray (target_any_smult)-1
    if target_all_smult is not None:
        target_all_s2 = np.asarray (target_all_smult)-1
    for i in range (las.nroots):
        idx[i] = idx[i] & ((max_charges is None) or np.all(charges[i]<=max_charges))
        idx[i] = idx[i] & ((min_charges is None) or np.all(charges[i]>=min_charges))
        idx[i] = idx[i] & ((max_smults is None) or np.all(smults[i]<=max_smults))
        idx[i] = idx[i] & ((min_smults is None) or np.all(smults[i]>=min_smults))
        s2 = smults[i]-1
        max_s2 = np.sum (s2) 
        min_s2 = 2*np.amax (s2) - max_s2
        if target_any_smult is not None:
            idx[i] = idx[i] & np.any (target_any_s2>=min_s2)
            idx[i] = idx[i] & np.any (target_any_s2<=max_s2)
        if target_all_smult is not None:
            idx[i] = idx[i] & np.all (target_all_s2>=min_s2)
            idx[i] = idx[i] & np.all (target_all_s2<=max_s2)
    weights = list (np.asarray (las.weights)[idx])
    if 1-np.sum (weights) > 1e-3: log.warn ("Filtered LAS spaces have less than unity weight!")
    return las.state_average (weights=weights, charges=charges[idx], spins=spins[idx],
                              smults=smults[idx], wfnsyms=wfnsyms[idx])

if __name__=='__main__':
    from mrh.tests.lasscf.c2h4n4_struct import structure as struct
    from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
    from mrh.my_pyscf.fci import csf_solver
    from pyscf import scf, mcscf
    from pyscf.tools import molden
    mol = struct (2.0, 2.0, '6-31g', symmetry=False)
    mol.spin = 8
    mol.verbose = logger.INFO
    mol.output = 'lassi_states.log'
    mol.build ()
    mf = scf.RHF (mol).run ()
    las = LASSCF (mf, (4,2,4), ((2,2),(1,1),(2,2)), spin_sub=(1,1,1))
    mo_coeff = las.localize_init_guess ([[0,1,2],[3,4,5,6],[7,8,9]])
    las.kernel (mo_coeff)
    elas0 = las.e_tot
    print ("LASSCF:", elas0)
    casdm1 = las.make_casdm1 ()
    no_coeff, no_ene, no_occ = las.canonicalize (natorb_casdm1=casdm1)[:3]
    molden.from_mo (las.mol, 'lassi_states.lasscf.molden', no_coeff, ene=no_ene, occ=no_occ)
    las2 = all_single_excitations (las)
    las2.lasci ()
    las2.dump_spaces ()
    e_roots, si = las2.lassi ()
    elas1 = e_roots[0]
    print ("LASSI(S):", elas1)
    from mrh.my_pyscf.mcscf import lassi
    casdm1 = lassi.root_make_rdm12s (las2, las2.ci, si, state=0)[0].sum (0)
    no_coeff, no_ene, no_occ = las.canonicalize (natorb_casdm1=casdm1)[:3]
    molden.from_mo (las.mol, 'lassi_states.lassis.molden', no_coeff, ene=no_ene, occ=no_occ)
    las3 = all_single_excitations (las2)
    las3.lasci ()
    las3.dump_spaces ()
    e_roots, si = las3.lassi ()
    elas2 = e_roots[0]
    print ("LASSI(SD):", elas2)
    casdm1 = lassi.root_make_rdm12s (las3, las3.ci, si, state=0)[0].sum (0)
    no_coeff, no_ene, no_occ = las.canonicalize (natorb_casdm1=casdm1)[:3]
    molden.from_mo (las.mol, 'lassi_states.lassisd.molden', no_coeff, ene=no_ene, occ=no_occ)
    las4 = all_single_excitations (las3)
    las4.lasci ()
    las4.dump_spaces ()
    e_roots, si = las4.lassi ()
    elas3 = e_roots[0]
    print ("LASSI(SDT):", elas3)
    casdm1 = lassi.root_make_rdm12s (las4, las4.ci, si, state=0)[0].sum (0)
    no_coeff, no_ene, no_occ = las.canonicalize (natorb_casdm1=casdm1)[:3]
    molden.from_mo (las.mol, 'lassi_states.lassisdt.molden', no_coeff, ene=no_ene, occ=no_occ)
    las5 = all_single_excitations (las4)
    las5.lasci ()
    las5.dump_spaces ()
    e_roots, si = las5.lassi ()
    elas4 = e_roots[0]
    print ("LASSI(SDTQ):", elas4)
    casdm1 = lassi.root_make_rdm12s (las5, las5.ci, si, state=0)[0].sum (0)
    no_coeff, no_ene, no_occ = las.canonicalize (natorb_casdm1=casdm1)[:3]
    molden.from_mo (las.mol, 'lassi_states.lassisdtq.molden', no_coeff, ene=no_ene, occ=no_occ)
    mc = mcscf.CASCI (mf, (10), (5,5)).set (fcisolver=csf_solver (mol, smult=1))
    mc.kernel (mo_coeff=las.mo_coeff)
    ecas = mc.e_tot
    print ("CASCI:", ecas)
    no_coeff, no_ci, no_occ = mc.cas_natorb ()
    molden.from_mo (las.mol, 'lassi_states.casci.molden', no_coeff, occ=no_occ)



