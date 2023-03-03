import numpy as np
from pyscf.fci.direct_spin1 import _unpack_nelec
from pyscf.lib import logger
from mrh.my_pyscf.fci.csfstring import CSFTransformer
from mrh.my_pyscf.fci.csfstring import ImpossibleSpinError
import itertools

class SingleLASState (object):
    def __init__(self, las, spins, smults, charges, nlas=None, nelelas=None, stdout=None,
                 verbose=None):
        if nlas is None: nlas = las.ncas_sub
        if nelelas is None: nelelas = [sum (_unpack_nelec (x)) for x in las.nelecas_sub]
        if stdout is None: stdout = las.stdout
        if verbose is None: verbose = las.verbose
        self.las = las
        self.nlas, self.nelelas = np.asarray (nlas), np.asarray (nelelas)
        self.nfrag = len (nlas)
        self.spins, self.smults = np.asarray (spins), np.asarray (smults)
        self.charges = np.asarray (charges)
        self.stdout, self.verbose = stdout, verbose
        
        self.nelec = self.nelelas - self.charges
        self.neleca = (self.nelec + self.spins) // 2
        self.nelecb = (self.nelec - self.spins) // 2
        self.nhole = 2*self.nlas - self.nelec 
        self.nholea = self.nlas - self.neleca
        self.nholeb = self.nlas - self.nelecb

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
        log.info ("c,m,s=[{},{},{}]->c,m,s=[{},{},{}]; {},{} CSFs".format (
            self.charges, self.spins, self.smults,
            charges, spins, smults,
            i_ncsf, a_ncsf))
        return SingleLASState (self.las, spins, smults, charges, nlas=self.nlas,
                               nelelas=self.nelelas, stdout=self.stdout, verbose=self.verbose)

    def get_singles (self):
        # move 1 alpha electron
        has_ea = np.where (self.neleca > 0)[0]
        has_ha = np.where (self.nholea > 0)[0]
        singles = []
        for i, a in itertools.product (has_ea, has_ha):
            if i==a: continue
            for si, sa in itertools.product (range (-1,2,2), repeat=2):
                try:
                    singles.append (self.get_single (i,a,0,si,sa))
                except ImpossibleSpinError as e:
                    pass
        # move 1 beta electron
        has_eb = np.where (self.nelecb > 0)[0]
        has_hb = np.where (self.nholeb > 0)[0]
        for i, a in itertools.product (has_ea, has_ha):
            if i==a: continue
            for si, sa in itertools.product (range (-1,2,2), repeat=2):
                try:
                    singles.append (self.get_single (i,a,1,si,sa))
                except ImpossibleSpinError as e:
                    pass
        return singles

def all_single_excitations (las):
    from mrh.my_pyscf.mcscf.lasci import get_state_info
    ref_states = [SingleLASState (las, m, s, c) for c,m,s,w in zip (*get_state_info (las))]
    new_states = []
    for ref_state in ref_states:
        new_states.extend (ref_state.get_singles ())
    return new_states


if __name__=='__main__':
    from mrh.tests.lasscf.c2h4n4_struct import structure as struct
    from pyscf import scf
    from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
    mol = struct (2.0, 2.0, '6-31g', symmetry=False)
    mol.verbose = logger.INFO
    mol.output = 'lassi_states.log'
    mol.build ()
    mf = scf.RHF (mol).run ()
    las = LASSCF (mf, (4,4), (4,4), spin_sub=(1,1))
    #las.state_average_(weights=[1.0/5.0,]*5,
    #    spins=[[0,0],[0,0],[2,-2],[-2,2],[2,2]],
    #    smults=[[1,1],[3,3],[3,3],[3,3],[3,3]])
    print (len (all_single_excitations (las)))






