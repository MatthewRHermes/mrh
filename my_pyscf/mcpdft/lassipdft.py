from pyscf import lib
import numpy as np
from mrh.my_pyscf.lassi import lassi
from pyscf.mcpdft.mcpdft import _mcscf_env
from mrh.my_pyscf.mcpdft.laspdft import _LASPDFT, _get_mcpdft_child_class, _instantiate

def make_casdm1s(filename, i):
    """
    This function stores the rdm1s for the given state 'i' in a tempfile
    """
    rdm1s_key = f'rdm1s_{i}'
    rdm1s = filename[rdm1s_key][:]
    rdm1s = np.array(rdm1s)
    return rdm1s


def make_casdm2s(filename, i):
    """
    This function stores the rdm2s for the given state 'i' in a tempfile
    """
    rdm2s_key = f'rdm2s_{i}'
    rdm2s = filename[rdm2s_key][:]
    rdm2s = np.array(rdm2s)
    return rdm2s


def get_mcpdft_child_class(mc, ot, states=None, _pdft_class=_LASPDFT, **kwargs):
    '''LASSI-PDFT: the LAS-PDFT class of mrh.my_pyscf.mcpdft.laspdft, evaluated on LASSI states'''
    LASPDFT = _get_mcpdft_child_class(mc, _pdft_class=_pdft_class)

    # These are attributes of the LASSI-PDFT class, not of the LAS class it is built on: setting
    # DoLASSI on the LAS class would make every later LAS-PDFT calculation on that class behave
    # like LASSI-PDFT (e.g., keep the per-state e_tot).
    class PDFT(LASPDFT):
        __doc__ = LASPDFT.__doc__
        DoLASSI = True
        states = None
        statlis = None
        rdmstmpfile = None

        def analyze(self, state=0, **kwargs):
            log = lib.logger.new_logger(self, self.verbose)
            log.warn("Analyze function is not yet defined for LAS-PDFT. Turning on the analyze function of LASSI")
            from mrh.my_pyscf.lassi.sitools import analyze
            return analyze(self, self.si, state=state, **kwargs)

        def get_energy_decomposition(self, *args, **kwargs):
            return super(LASPDFT, self).get_energy_decomposition(*args, **kwargs)

        '''
        The cost of the RDM build is similar to LASSI diagonalization step. Therefore,
        calling it 2n time for n-states becomes prohibitively expensive. One alternative
        can be just call it once and store all the generated casdm1 and casdm2 and later on
        just call a reader function which will read the rdms from this temp file.
        '''

        def _store_rdms(self):
            # MRH: I made it loop over blocks of states to handle the O(N^5) memory cost
            # If there's enough memory it'll still do them all at once
            log = lib.logger.new_logger(self, self.verbose)
            safety_factor = 1.3
            mem_per_state = safety_factor * 8 * (2 * (self.ncas ** 2) + 4 * (self.ncas ** 4)) / 1e6
            current_mem = lib.current_memory()[0]

            if current_mem > self.max_memory:
                log.warn("Current memory usage (%d MB) exceeds maximum memory (%d MB)",
                         current_mem, self.max_memory)
                nblk = 1
            else:
                nblk = max(1, int((self.max_memory - current_mem) / mem_per_state) - 1)

            log.debug('_store_rdms: looping over %d states at a time of %d total', len(self.states),
                      nblk)

            for i in range(0, len(self.states), nblk):
                j = min(i + nblk, len(self.states))

                rdm1s, rdm2s = lassi.root_make_rdm12s(self, self.ci, self.si,
                                                      state=self.states[i:j])

                if len(self.states[i:j]) == 1:
                    rdm1s = [rdm1s]
                    rdm2s = [rdm2s]

                for k in range(i, j):
                    stateno = self.states[k]
                    rdm1s_dname = f'rdm1s_{stateno}'
                    self.rdmstmpfile.create_dataset(rdm1s_dname, data=rdm1s[k])
                    rdm2s_dname = f'rdm2s_{stateno}'
                    self.rdmstmpfile.create_dataset(rdm2s_dname, data=rdm2s[k])

                rdm1s = rdm2s = None

        def make_one_casdm1s(self, ci=None, state=0, **kwargs):
            rdmstmpfile = self.rdmstmpfile
            return make_casdm1s(rdmstmpfile, self.states[state])

        def make_one_casdm2(self, ci=None, state=0, **kwargs):
            rdmstmpfile = self.rdmstmpfile
            return make_casdm2s(rdmstmpfile, self.states[state]).sum((0, 3))

        def optimize_mcscf_(self, mo_coeff=None, ci0=None, **kwargs):
            '''Optimize the MC-SCF wave function underlying an MC-PDFT calculation.
            Has the same calling signature as the parent kernel method. '''

            with _mcscf_env(self):
                self.statlis = [x for x in range(len(self.e_roots))]  # LASSI-LPDFT
                if self.states is None:
                    self.states = list(range(len(self.e_roots)))
                    self.fcisolver.nroots = len(self.e_roots)
                else:
                    self.fcisolver.nroots = len(self.states)

                self.rdmstmpfile = lib.H5TmpFile ()
                self._store_rdms()

            self.e_mcscf = self.e_roots[self.states]  # To be consistent with PySCF

        def __del__(self):
            if callable (getattr (self.rdmstmpfile, 'close', None)):
                self.rdmstmpfile.close ()

    pdft = _instantiate(PDFT, mc, ot, **kwargs)
    if states is not None: pdft.states = states
    return pdft
