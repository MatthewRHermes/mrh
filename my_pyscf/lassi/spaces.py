import numpy as np
from scipy import linalg
from pyscf.fci.direct_spin1 import _unpack_nelec
from pyscf.fci import cistring
from pyscf.lib import logger
from pyscf.lo.orth import vec_lowdin
from pyscf import symm, __config__
from mrh.my_pyscf.lib.logger import select_log_printer
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.fci.spin_op import contract_sdown, contract_sup, mdown, mup
from mrh.my_pyscf.fci.csfstring import CSFTransformer
from mrh.my_pyscf.fci.csfstring import ImpossibleSpinError
from mrh.my_pyscf.mcscf.productstate import ImpureProductStateFCISolver
import itertools

LINDEP_THRESH = getattr (__config__, 'lassi_lindep_thresh', 1.0e-5)

class SingleLASRootspace (object):
    def __init__(self, las, spins, smults, charges, weight, nlas=None, nelelas=None, stdout=None,
                 verbose=None, ci=None, fragsym=None, energy_tot=0):
        if nlas is None: nlas = las.ncas_sub
        if nelelas is None: nelelas = [sum (_unpack_nelec (x)) for x in las.nelecas_sub]
        if stdout is None: stdout = las.stdout
        if verbose is None: verbose = las.verbose
        if fragsym is None: fragsym = np.zeros_like (len(nlas))
        self.las = las
        self.nlas, self.nelelas = np.asarray (nlas), np.asarray (nelelas)
        self.nfrag = len (nlas)
        self.spins, self.smults = np.asarray (spins), np.asarray (smults)
        self.charges = np.asarray (charges)
        self.weight = weight
        self.stdout, self.verbose = stdout, verbose
        self.ci = ci
        self.fragsym = fragsym

        self.nelec = self.nelelas - self.charges
        self.neleca = (self.nelec + self.spins) // 2
        self.nelecb = (self.nelec - self.spins) // 2
        self.nhole = 2*self.nlas - self.nelec 
        self.nholea = self.nlas - self.neleca
        self.nholeb = self.nlas - self.nelecb

        # "u", "d": like "a", "b", but presuming spins+1==smults everywhere
        self.nelecu = (self.nelec + (self.smults-1)) // 2
        self.nelecd = (self.nelec - (self.smults-1)) // 2
        self.nholeu = self.nlas - self.nelecu
        self.nholed = self.nlas - self.nelecd

        self.energy_tot = energy_tot

        self.entmap = tuple ()

    def __eq__(self, other):
        if self.nfrag != other.nfrag: return False
        return (np.all (self.spins==other.spins) and 
                np.all (self.smults==other.smults) and
                np.all (self.charges==other.charges))

    def __hash__(self):
        return hash (tuple ([self.nfrag,] + list (self.spins) + list (self.smults)
                            + list (self.charges) + list (self.entmap)))

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

    def get_single_any_m (self, i, a, dsi, dsa, ci_i=None, ci_a=None):
        mi, ma = self.spins[i], self.spins[a]
        si, sa = self.smults[i]+dsi, self.smults[a]+dsa
        i_a_pos = self.neleca[i]>0 and abs(mi-1)<si
        i_b_pos = self.nelecb[i]>0 and abs(mi+1)<si
        a_a_pos = self.nholea[a]>0 and abs(ma+1)<sa
        a_b_pos = self.nholeb[a]>0 and abs(ma-1)<sa
        ofrags = np.ones (self.nfrag, dtype=bool)
        ofrags[i] = ofrags[a] = False
        max_up = sum((self.smults-1-self.spins)[ofrags])//2
        max_dn = sum((self.spins+self.smults-1)[ofrags])//2
        p = None
        if i_a_pos and a_a_pos:
            sp = self.get_single (i, a, 0, dsi, dsa)
        elif i_b_pos and a_b_pos:
            sp = self.get_single (i, a, 1, dsi, dsa)
        elif i_a_pos and a_b_pos and max_up:
            p = np.where (ofrags & (self.spins<self.smults))[0][0]
            dsp = 1 if self.nholeu[p]>0 else -1
            sp = self.get_single (i, p, 0, dsi, dsp).get_single (p, a, 1, -dsp, dsa)
        elif i_b_pos and a_a_pos and max_dn:
            p = np.where (ofrags & (self.spins>-self.smults))[0][0]
            dsp = 1 if self.nholeu[p]>0 else -1
            sp = self.get_single (i, p, 1, dsi, dsp).get_single (p, a, 0, -dsp, dsa)
        else:
            raise ImpossibleSpinError ((
                "Can't figure out legal excitation (norb={}, neleca={}, nelecb={}, smults={}, "
                "i = {}, a = {}, dsi = {}, dsa = {} {} {} {} {} {} {}").format (self.nlas, self.neleca, self.nelecb,
                self.smults, i, a, dsi, dsa, i_a_pos, i_b_pos, a_a_pos, a_b_pos, max_up, max_dn)
            )
        if self.has_ci () and ci_i is not None and ci_a is not None:
            sp.ci = [x for x in self.ci]
            sp.ci[i] = mdown (ci_i, sp.nlas[i], (sp.neleca[i],sp.nelecb[i]), sp.smults[i])
            sp.ci[a] = mdown (ci_a, sp.nlas[a], (sp.neleca[a],sp.nelecb[a]), sp.smults[a])
            if p is not None:
                nelec0 = (self.neleca[p],self.nelecb[p])
                nelec1 = (sp.neleca[p],sp.nelecb[p])
                norb, smult = self.nlas[p], self.smults[p]
                sp.ci[p] = mdown (mup (self.ci[p], norb, nelec0, smult), norb, nelec1, smult)
        sp.set_entmap_(self, ignore_m=True)
        return sp

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
            sp = SingleLASRootspace (self.las, spins, self.smults, self.charges, 0, nlas=self.nlas,
                                     nelelas=self.nelelas, stdout=self.stdout, verbose=self.verbose)
            sp.entmap = self.entmap
            yield sp

    def has_ci (self):
        if self.ci is None: return False
        return all ([c is not None for c in self.ci])

    def get_ci_szrot (self, ifrags=None):
        '''Generate the sets of CI vectors in which each vector for each fragment
        has the sz axis rotated in all possible ways.

        Kwargs:
            ifrags: list of integers
                Optionally restrict ci_sz to particular fragments identified by ifrags

        Returns:
            ci_sz: list of dict of type {integer: ndarray}
                dict keys are integerified "spin" quantum numbers; i.e., neleca-nelecb.
                dict vals are the corresponding CI vectors
        '''
        ci_sz = []
        ndet = self.get_ndet ()
        if ifrags is None: ifrags = range (self.nfrag)
        for ifrag in ifrags:
            norb, sz, ci = self.nlas[ifrag], self.spins[ifrag], self.ci[ifrag]
            ndeta, ndetb = ndet[ifrag]
            nelec = self.neleca[ifrag], self.nelecb[ifrag]
            smult = self.smults[ifrag]
            ci_sz_ = {sz: ci}
            ci1 = np.asarray (ci).reshape (-1, ndeta, ndetb)
            nvecs = ci1.shape[0]
            nelec1 = nelec
            for sz1 in range (sz-2, -(1+smult), -2):
                ci1 = [contract_sdown (c, norb, nelec1) for c in ci1]
                nelec1 = nelec1[0]-1, nelec1[1]+1
                if nvecs==1: ci_sz_[sz1] = ci1[0]
                else: ci_sz_[sz1] = np.asarray (ci1)
            ci1 = np.asarray (ci).reshape (nvecs, ndeta, ndetb)
            nelec1 = nelec
            for sz1 in range (sz+2, (1+smult), 2):
                ci1 = [contract_sup (c, norb, nelec1) for c in ci1]
                nelec1 = nelec1[0]+1, nelec1[1]-1
                if nvecs==1: ci_sz_[sz1] = ci1[0]
                else: ci_sz_[sz1] = np.asarray (ci1)
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

    def describe_single_excitation (self, other):
        if not self.is_single_excitation_of (other): return None
        src_frag = np.where ((self.nelec-other.nelec)==-1)[0][0]
        dest_frag = np.where ((self.nelec-other.nelec)==1)[0][0]
        e_spin = 'a' if np.any (self.neleca!=other.neleca) else 'b'
        src_ds = 'u' if self.smults[src_frag]>other.smults[src_frag] else 'd'
        dest_ds = 'u' if self.smults[dest_frag]>other.smults[dest_frag] else 'd'
        lroots_s = min (other.nelecu[src_frag], other.nholed[dest_frag])
        return src_frag, dest_frag, e_spin, src_ds, dest_ds, lroots_s

    def single_excitation_key (self, other):
        i, a, _, si, sa, _ = self.describe_single_excitation (other)
        spin = (2 * int (si=='u')) + int (sa=='u')
        return i, a, spin

    def set_entmap_(self, ref, ignore_m=False):
        idx = np.where (self.excited_fragments (ref, ignore_m=ignore_m))[0]
        idx = tuple (set (idx))
        self.entmap = tuple ((idx,))
        #self.entmap[:,:] = 0
        #for i, j in itertools.combinations (idx, 2):
        #    self.entmap[i,j] = self.entmap[j,i] = 1

    def single_excitation_description_string (self, other):
        src, dest, e_spin, src_ds, dest_ds, lroots_s = self.describe_single_excitation (other)
        fmt_str = '{:d}({:s}) --{:s}--> {:d}({:s})'
        return fmt_str.format (src, src_ds, e_spin, dest, dest_ds)

    def compute_single_excitation_lroots (self, ref):
        if isinstance (ref, (list, tuple)):
            lroots = np.array ([self.compute_single_excitation_lroots (r) for r in ref])
            return np.amax (lroots)
        assert (self.is_single_excitation_of (ref))
        return self.describe_single_excitation (ref)[5]

    def is_spin_shuffle_of (self, other):
        if np.any (self.nelec != other.nelec): return False
        if np.any (self.smults != other.smults): return False
        return self.spins.sum () == other.spins.sum ()

    def get_spin_shuffle_civecs (self, other):
        assert (self.is_spin_shuffle_of (other) and other.has_ci ())
        ci_sz = other.get_ci_szrot ()
        return [ci_sz[ifrag][self.spins[ifrag]] for ifrag in range (self.nfrag)]

    def excited_fragments (self, other, ignore_m=False):
        if other is None: return np.ones (self.nfrag, dtype=bool)
        dneleca = self.neleca - other.neleca
        dnelecb = self.nelecb - other.nelecb
        dsmults = self.smults - other.smults
        if ignore_m:
            dneleca += dnelecb
            dnelecb[:] = 0
        idx_same = (dneleca==0) & (dnelecb==0) & (dsmults==0)
        return ~idx_same

    def get_lroots (self):
        if not self.has_ci (): return None
        lroots = []
        for c, n in zip (self.ci, self.get_ndet ()):
            c = np.asarray (c).reshape (-1, n[0], n[1])
            lroots.append (c.shape[0])
        return lroots

    def table_printlog (self, lroots=None, tverbose=logger.INFO):
        if lroots is None: lroots = self.get_lroots ()
        log = logger.new_logger (self, self.verbose)
        printer = select_log_printer (log, tverbose=tverbose)
        fmt_str = " {:4s}  {:>11s}  {:>4s}  {:>3s}"
        header = fmt_str.format ("Frag", "Nelec,Norb", "2S+1", "Ir")
        fmt_str = " {:4d}  {:>11s}  {:>4d}  {:>3s}"
        if lroots is not None:
            header += '  Nroots'
            fmt_str += '  {:>6d}'
        printer (header)
        for ifrag in range (self.nfrag):
            na, nb = self.neleca[ifrag], self.nelecb[ifrag]
            sm, no = self.smults[ifrag], self.nlas[ifrag]
            irid = 0 # TODO: symmetry
            nelec_norb = '{}a+{}b,{}o'.format (na,nb,no)
            irname = symm.irrep_id2name (self.las.mol.groupname, irid)
            row = [ifrag, nelec_norb, sm, irname]
            if lroots is not None: row += [lroots[ifrag]]
            printer (fmt_str.format (*row))

    def single_fragment_spin_change (self, ifrag, new_smult, new_spin, ci=None):
        smults1 = self.smults.copy ()
        spins1 = self.spins.copy ()
        smults1[ifrag] = new_smult
        spins1[ifrag] = new_spin
        ci1 = None
        if ci is not None:
            ci1 = [c for c in self.ci]
            ci1[ifrag] = ci
        sp = SingleLASRootspace (self.las, spins1, smults1, self.charges, 0, nlas=self.nlas,
                                 nelelas=self.nelelas, stdout=self.stdout, verbose=self.verbose,
                                 ci=ci1)
        sp.entmap = self.entmap
        assert (ci is sp.ci[ifrag])
        return sp

    def is_orthogonal_by_smult (self, other):
        if isinstance (other, (list, tuple)):
            return [self.is_orthogonal_by_smult (o) for o in other]
        s2_self = self.smults-1
        max_self = np.sum (s2_self) 
        min_self = 2*np.amax (s2_self) - max_self
        s2_other = other.smults-1
        max_other = np.sum (s2_other) 
        min_other = 2*np.amax (s2_other) - max_other
        return (max_self < min_other) or (max_other < min_self)

    def get_fcisolvers (self):
        fcisolvers = []
        for ifrag in range (self.nfrag):
            solver = csf_solver (self.las.mol, smult=self.smults[ifrag])
            solver.nelec = (self.neleca[ifrag],self.nelecb[ifrag])
            solver.norb = self.nlas[ifrag]
            solver.spin = self.spins[ifrag]
            solver.check_transformer_cache ()
            fcisolvers.append (solver)
        return fcisolvers

    def get_product_state_solver (self, lroots=None, lweights='gs'):
        fcisolvers = self.get_fcisolvers ()
        if lroots is None: lroots = self.get_lroots ()
        lw = [np.zeros (l) for l in lroots]
        if 'gs' in lweights.lower ():
            for l in lw: l[0] = 1.0
        elif 'sa' in lweights.lower ():
            for l in lw: l[:] = 1.0/len (l)
        else:
            raise RuntimeError ('valid lweights are "gs" and "sa"')
        lweights=lw
        return ImpureProductStateFCISolver (fcisolvers, stdout=self.stdout, lweights=lweights, 
                                            verbose=self.verbose)

    def merge_(self, other, ref=None, lindep_thresh=LINDEP_THRESH):
        idx = self.excited_fragments (ref)
        ndet = self.get_ndet ()
        for ifrag in np.where (idx)[0]:
            self.ci[ifrag] = np.append (
                np.asarray (self.ci[ifrag]).reshape (-1, ndet[ifrag][0], ndet[ifrag][1]),
                np.asarray (other.ci[ifrag]).reshape (-1, ndet[ifrag][0], ndet[ifrag][1]),
                axis=0
            )
            ovlp = np.tensordot (self.ci[ifrag].conj (), self.ci[ifrag],
                                 axes=((1,2),(1,2)))
            if linalg.det (ovlp) < LINDEP_THRESH:
                evals, evecs = linalg.eigh (ovlp)
                idx = evals>LINDEP_THRESH
                evals, evecs = evals[idx], evecs[:,idx]
                evecs /= np.sqrt (evals)[None,:]
                self.ci[ifrag] = np.tensordot (evecs.T, self.ci[ifrag], axes=1)

    def get_wfnsym (self):
        return np.bitwise_xor.reduce (self.fragsym)

    def get_s2_exptval (self):
        s = (self.smults - 1) / 2
        m = self.spins / 2
        s2 = np.multiply.outer (m, m)
        s2[np.diag_indices_from (s2)] = s*(s+1)
        return s2.sum ()

def orthogonal_excitations (exc1, exc2, ref):
    if exc1.nfrag != ref.nfrag: return False
    if exc2.nfrag != ref.nfrag: return False
    idx1 = exc1.excited_fragments (ref)
    if not np.count_nonzero (idx1): return False
    idx2 = exc2.excited_fragments (ref)
    if not np.count_nonzero (idx2): return False
    if np.count_nonzero (idx1 & idx2): return False
    return True

def combine_orthogonal_excitations (exc1, exc2, ref):
    nfrag = ref.nfrag
    spins = exc1.spins.copy ()
    smults = exc1.smults.copy ()
    charges = exc1.charges.copy ()
    idx2 = exc2.excited_fragments (ref)
    spins[idx2] = exc2.spins[idx2]
    smults[idx2] = exc2.smults[idx2]
    charges[idx2] = exc2.charges[idx2]
    ci = None
    if exc1.has_ci () and exc2.has_ci ():
        ci = [exc2.ci[ifrag] if idx2[ifrag] else exc1.ci[ifrag] for ifrag in range (nfrag)]
    product = SingleLASRootspace (
        ref.las, spins, smults, charges, 0, ci=ci,
        nlas=ref.nlas, nelelas=ref.nelelas, stdout=ref.stdout, verbose=ref.verbose
    )
    product.entmap = tuple (set (exc1.entmap + exc2.entmap))
    #assert (np.amax (product.entmap) < 2)
    assert (len (product.entmap) == len (set (product.entmap)))
    for ifrag in range (nfrag):
        assert ((product.ci[ifrag] is exc1.ci[ifrag]) or
                (product.ci[ifrag] is exc2.ci[ifrag]) or
                (product.ci[ifrag] is ref.ci[ifrag]))
    return product

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
    log.info ('Built {} singly-excited LAS states from {} reference LAS states'.format (
        len (all_states) - len (ref_states), len (ref_states)))
    if len (all_states) == len (ref_states):
        log.warn (("%d reference LAS states exhaust current active space specifications; "
                   "no singly-excited states could be constructed"), len (ref_states))
    weights = [state.weight for state in all_states]
    charges = [state.charges for state in all_states]
    spins = [state.spins for state in all_states]
    smults = [state.smults for state in all_states]
    #wfnsyms = [state.wfnsyms for state in all_states]
    return las.state_average (weights=weights, charges=charges, spins=spins, smults=smults)

def spin_shuffle (las, verbose=None, equal_weights=False):
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
    all_states = _spin_shuffle (ref_states, equal_weights=equal_weights)
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

def _spin_shuffle (ref_spaces, equal_weights=False):
    '''The same as spin_shuffle, but the inputs and outputs are space lists rather than LASSCF
    instances and no logging is done.'''
    seen = set (ref_spaces)
    all_spaces = [space for space in ref_spaces]
    for ref_space in ref_spaces:
        for new_space in ref_space.gen_spin_shuffles ():
            if not new_space in seen:
                all_spaces.append (new_space)
                seen.add (new_space)
    if equal_weights:
        w = 1.0/len(all_spaces)
        for space in all_spaces: space.weight = w
    return all_spaces

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
    spaces = _spin_shuffle_ci_(spaces)
    ci = [[space.ci[ifrag] for space in spaces] for ifrag in range (las.nfrags)]
    return ci

def _spin_shuffle_ci_(spaces):
    old_ci_sz = []
    old_idx = []
    new_idx = []
    nfrag = spaces[0].nfrag
    for ix, space in enumerate (spaces):
        if space.has_ci ():
            old_idx.append (ix)
            old_ci_sz.append (space.get_ci_szrot ())
        else:
            new_idx.append (ix)
            space.ci = [None for ifrag in range (space.nfrag)]
    def is_spin_shuffle_ref (sp1, sp2):
        return (np.all (sp1.charges==sp2.charges) and
                np.all (sp1.smults==sp2.smults) and
                sp1.entmap==sp2.entmap)
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
                spaces[ix].ci[ifrag] = None
                continue
            lroots, ndeti = ci_ix[ifrag].shape[0], ndet[ifrag]
            if lroots > 1:
                c = (ci_ix[ifrag].reshape (lroots, ndeti[0]*ndeti[1])).T
                ovlp = c.conj ().T @ c
                w, v = linalg.eigh (ovlp)
                idx = w>1e-8
                v = v[:,idx] / np.sqrt (w[idx])[None,:]
                c = (c @ v).T
                ci_ix[ifrag] = c.reshape (-1, ndeti[0], ndeti[1])
            spaces[ix].ci[ifrag] = ci_ix[ifrag]
    return spaces

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

def list_spaces (las):
    from mrh.my_pyscf.mcscf.lasci import get_space_info
    spaces = [SingleLASRootspace (las, m, s, c, las.weights[ix], ci=[c[ix] for c in las.ci],
                                  fragsym=w)
              for ix, (c, m, s, w) in enumerate (zip (*get_space_info (las)))]
    for e_state, space in zip (las.e_states, spaces):
        space.energy_tot = e_state
    return spaces

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



