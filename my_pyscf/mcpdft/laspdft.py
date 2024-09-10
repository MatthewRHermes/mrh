from pyscf import ao2mo, lib
from pyscf.mcscf.addons import StateAverageMCSCFSolver
import numpy as np
from mrh.my_pyscf.lassi import lassi
import h5py
import tempfile
from pyscf.mcpdft.otfnal import transfnal, get_transfnal
from pyscf.mcpdft.mcpdft import _get_e_decomp

try:
    from pyscf.mcpdft.mcpdft import _PDFT, _mcscf_env
except ImportError:
    msg = "For performing LASPDFT, you will require pyscf-forge.\n" + \
          "pyscf-forge can be found at : https://github.com/pyscf/pyscf-forge"
    raise ImportError(msg)


def make_casdm1s(filename, i):
    """
    This function stores the rdm1s for the given state 'i' in a tempfile
    """
    with h5py.File(filename, 'r') as f:
        rdm1s_key = f'rdm1s_{i}'
        rdm1s = f[rdm1s_key][:]
        rdm1s = np.array(rdm1s)
    return rdm1s


def make_casdm2s(filename, i):
    """
    This function stores the rdm2s for the given state 'i' in a tempfile
    """
    with h5py.File(filename, 'r') as f:
        rdm2s_key = f'rdm2s_{i}'
        rdm2s = f[rdm2s_key][:]
        rdm2s = np.array(rdm2s)
    return rdm2s


def get_energy_decomposition(mc, mo_coeff=None, ci=None, ot=None, otxc=None,
                             grids_level=None, grids_attr=None,
                             split_x_c=None, verbose=None):
    """
    I have to include this function to do the EDA for selected roots only.
    """
    if verbose is None: verbose = mc.verbose
    log = lib.logger.new_logger(mc, verbose)
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci
    if grids_attr is None: grids_attr = {}
    if grids_level is not None: grids_attr['level'] = grids_level
    if len(grids_attr) or (otxc is not None):
        old_ot = ot if (ot is not None) else mc.otfnal
        old_grids = old_ot.grids
        if otxc is None: otxc = old_ot.otxc
        new_ot = get_transfnal(mc.mol, otxc)
        new_ot.grids.__dict__.update(old_grids.__dict__)
        new_ot.grids.__dict__.update(**grids_attr)
        ot = new_ot
    if ot is None: ot = mc.otfnal
    if split_x_c is None:
        split_x_c = True
        log.warn('Currently, split_x_c in get_energy_decomposition defaults to '
                 'True.\nThis default will change to False in the near future.')
    hyb_x, hyb_c = ot._numint.hybrid_coeff(ot.otxc)
    if hyb_x > 1e-10 or hyb_c > 1e-10:
        raise NotImplementedError("Decomp for hybrid PDFT fnals")
    if not isinstance(ot, transfnal):
        raise NotImplementedError("Decomp for non-translated PDFT fnals")

    if split_x_c:
        ot = list(ot.split_x_c())
    else:
        ot = [ot, ]
    e_nuc = mc._scf.energy_nuc()

    nroots = getattr(mc.fcisolver, 'nroots', 1)
    if not lib.issequence (nroots):
        nroots = list(range(nroots))

    if lib.issequence (nroots) and len(nroots) > 1:
        e_1e, e_coul, e_otxc, e_ncwfn = [], [], [], []
        for state in nroots:
            row = _get_e_decomp(mc, mo_coeff=mo_coeff,ci=mc.ci,ot=ot,state=state)
            e_1e.append(row[0])
            e_coul.append(row[1])
            e_otxc.append(row[2])
            e_ncwfn.append(row[3])
        e_otxc = [[e[i] for e in e_otxc] for i in range(len(e_otxc[0]))]
    else:
        e_1e, e_coul, e_otxc, e_ncwfn = _get_e_decomp(mc,mo_coeff=mo_coeff,ci=mc.ci,ot=ot,state=nroots[0])
    if split_x_c:
        e_otx, e_otc = e_otxc
        return e_nuc, e_1e, e_coul, e_otx, e_otc, e_ncwfn
    else:
        return e_nuc, e_1e, e_coul, e_otxc[0], e_ncwfn


class _LASPDFT(_PDFT):
    'MC-PDFT energy for a LASSCF wavefunction'

    def get_h2eff(self, mo_coeff=None):
        'Compute the active space two-particle Hamiltonian.'
        ncore = self.ncore
        ncas = self.ncas
        nocc = ncore + ncas
        if mo_coeff is None:
            mo_coeff = self.mo_coeff[:, ncore:nocc]
        elif mo_coeff.shape[1] != ncas:
            mo_coeff = mo_coeff[:, ncore:nocc]

        if getattr(self._scf, '_eri', None) is not None:
            eri = ao2mo.full(self._scf._eri, mo_coeff,
                             max_memory=self.max_memory)
        elif self.with_df:
            eri = self.with_df.ao2mo(mo_coeff)

        else:
            eri = ao2mo.full(self.mol, mo_coeff, verbose=self.verbose,
                             max_memory=self.max_memory)
        return eri

    def compute_pdft_energy_(self, mo_coeff=None, ci=None, ot=None, otxc=None,
                             grids_level=None, grids_attr=None, **kwargs):
        '''Compute the MC-PDFT energy(ies) (and update stored data)
        with the MC-SCF wave function fixed. '''
        '''
        Instead of finding the energies of all the states, this can allow
        to take state number for which you want to add the PDFT corrections
        '''
        if mo_coeff is not None: self.mo_coeff = mo_coeff
        if ci is not None: self.ci = ci
        if ot is not None: self.otfnal = ot
        if otxc is not None: self.otxc = otxc
        if grids_attr is None: grids_attr = {}
        if grids_level is not None: grids_attr['level'] = grids_level
        if len(grids_attr): self.grids.__dict__.update(**grids_attr)
        nroots = getattr(self.fcisolver, 'nroots', 1)
        if isinstance(nroots, list):
            epdft = [self.energy_tot(mo_coeff=self.mo_coeff, ci=self.ci, state=ix,
                                     logger_tag='MC-PDFT state {}'.format(ix))
                     for ix in nroots]
        else:
            epdft = [self.energy_tot(mo_coeff=self.mo_coeff, ci=self.ci, state=ix,
                                     logger_tag='MC-PDFT state {}'.format(ix))
                     for ix in range(nroots)]

        self.e_ot = np.array([e_ot for e_tot, e_ot in epdft])

        if isinstance(self, StateAverageMCSCFSolver):
            e_states = [e_tot for e_tot, e_ot in epdft]
            try:
                self.e_states = e_states
            except AttributeError as e:
                self.fcisolver.e_states = e_states
                assert (self.e_states is e_states), str(e)
            # TODO: redesign this. MC-SCF e_states is stapled to
            # fcisolver.e_states, but I don't want MS-PDFT to be
            # because that makes no sense
            self.e_tot = np.dot(e_states, self.weights)
            e_states = self.e_states
        elif (len(nroots) > 1 if isinstance(nroots, list) else nroots > 1):
            self.e_tot = np.array([e_tot for e_tot, e_ot in epdft])
            e_states = self.e_tot
        else:  # nroots==1 not StateAverage class
            self.e_tot, self.e_ot = epdft[0]
            e_states = [self.e_tot]
        return self.e_tot, self.e_ot, e_states

    def multi_state(self, method='Lin'):
        if method.upper() == "LIN":
            from mrh.my_pyscf.mcpdft._lpdft import linear_multi_state
            return linear_multi_state(self)
        else:
            raise NotImplementedError(f"StateAverageMix not available for {method}")


def get_mcpdft_child_class(mc, ot, DoLASSI=False, states=None, **kwargs):
    mc_doc = (mc.__class__.__doc__ or 'No docstring for MC-SCF parent method')

    class PDFT(_LASPDFT, mc.__class__):
        __doc__ = mc_doc + '\n\n' + _LASPDFT.__doc__
        _mc_class = mc.__class__
        setattr(_mc_class, 'DoLASSI', None)
        setattr(_mc_class, 'states', None)
        setattr(_mc_class, 'statlis', None)
        setattr(_mc_class, 'rdmstmpfile', None)

        def get_h2eff(self, mo_coeff=None):
            if self._in_mcscf_env:
                return mc.__class__.get_h2eff(self, mo_coeff=mo_coeff)
            else:
                return _LASPDFT.get_h2eff(self, mo_coeff=mo_coeff)

        def compute_pdft_energy_(self, mo_coeff=None, ci=None, ot=None, otxc=None,
                                 grids_level=None, grids_attr=None, states=states, **kwargs):
            return _LASPDFT.compute_pdft_energy_(self, mo_coeff=mo_coeff, ci=ci, ot=ot, otxc=otxc,
                                                 grids_level=grids_level, grids_attr=grids_attr, **kwargs)

        def get_energy_decomposition(self, **kwargs):
            raise NotImplementedError('EDA is not yet defined for LAS-PDFT')

        def multi_state(self, **kwargs):
            """
            In future will have to change this to consider the modal space selection, weights...
            """
            assert self.DoLASSI, "multi_state is only defined for post LAS methods"
            return _LASPDFT.multi_state(self, **kwargs)

        multi_state_mix = multi_state

        if DoLASSI:
            _mc_class.DoLASSI = True
            _mc_class.rdmstmpfile = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)

            def analyze(self, state=0, **kwargs):
                log = lib.logger.new_logger(self, self.verbose)
                log.warn("Analyze function is not yet defined for LAS-PDFT. Turning on the analyze function of LASSI")
                from mrh.my_pyscf.lassi.sitools import analyze
                return analyze(self, self.si, state=state, **kwargs)

            def get_energy_decomposition(self, mo_coeff=None, ci=None, ot=None, otxc=None, grids_level=None,
                                         grids_attr=None, split_x_c=None, verbose=None):
                return get_energy_decomposition(self, mo_coeff=mo_coeff, ci=ci, ot=ot, otxc=otxc,
                                                grids_level=grids_level, grids_attr=grids_attr,
                                                split_x_c=split_x_c, verbose=verbose)
        else:
            _mc_class.DoLASSI = False
            if mc.ci is not None:
                mc.fcisolver.nroots = mc.nroots
            else:
                mc.fcisolver.nroots = 1

            def analyze(self):
                raise NotImplementedError('Analyze function is not yet implemented for LAS-PDFT')

            def get_energy_decomposition(self, **kwargs):
                raise NotImplementedError('EDA is not yet implemented for LAS-PDFT')

        if states is not None: _mc_class.states = states

        if _mc_class.DoLASSI:

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

                rdmstmpfile = self.rdmstmpfile
                with h5py.File(rdmstmpfile, 'a') as f:
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
                            f.create_dataset(rdm1s_dname, data=rdm1s[k])
                            rdm2s_dname = f'rdm2s_{stateno}'
                            f.create_dataset(rdm2s_dname, data=rdm2s[k])

                        rdm1s = rdm2s = None

            def make_one_casdm1s(self, ci=None, state=0, **kwargs):
                rdmstmpfile = self.rdmstmpfile
                return make_casdm1s(rdmstmpfile, state)

            def make_one_casdm2(self, ci=None, state=0, **kwargs):
                rdmstmpfile = self.rdmstmpfile
                return make_casdm2s(rdmstmpfile, state).sum((0, 3))

        else:
            make_one_casdm1s = mc.__class__.state_make_casdm1s
            make_one_casdm2 = mc.__class__.state_make_casdm2

        # TODO: compatibility with MC-PDFT checkpoint dumping
        dump_chk = mc.__class__.dump_chk

        # TODO: in pyscf-forge/pyscf/mcpdft/mcpdft.py::optimize_mcscf_, generalize the number
        # of return arguments. Then the redefinition below will be unnecessary. 
        def optimize_mcscf_(self, mo_coeff=None, ci0=None, **kwargs):
            '''Optimize the MC-SCF wave function underlying an MC-PDFT calculation.
            Has the same calling signature as the parent kernel method. '''

            with _mcscf_env(self):
                if self.DoLASSI:
                    self.statlis = [x for x in range(len(self.e_roots))]  # LASSI-LPDFT
                    if self.states is None:
                        self.fcisolver.nroots = len(self.e_roots)
                        self.states = list(range(len(self.e_roots)))
                    else:
                        self.fcisolver.nroots = self.states

                    self._store_rdms()
                else:
                    self.e_mcscf, self.e_cas, self.ci, self.mo_coeff, self.mo_energy = \
                        self._mc_class.kernel(self, mo_coeff, ci0=ci0, **kwargs)[:-2]
                    self.fcisolver.nroots = self.nroots

            if self.DoLASSI:
                self.e_mcscf = self.e_roots[self.states]  # To be consistent with PySCF

    pdft = PDFT(mc._scf, mc.ncas_sub, mc.nelecas_sub, my_ot=ot, **kwargs)
    _keys = pdft._keys.copy()
    pdft.__dict__.update(mc.__dict__)
    pdft._keys = pdft._keys.union(_keys)
    return pdft
