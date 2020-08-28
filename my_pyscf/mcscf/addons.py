import numpy as np
from scipy import linalg, special
from pyscf.lib import logger, temporary_env
from pyscf.mcscf.addons import StateAverageMCSCFSolver, StateAverageMixFCISolver, state_average_mix
from pyscf.mcscf.addons import StateAverageMixFCISolver_state_args as _state_arg
from pyscf.mcscf.addons import StateAverageMixFCISolver_solver_args as _solver_arg

class StateAverageNMixFCISolver (StateAverageMixFCISolver):
    pass

def get_sanmix_fcisolver (samix_fcisolver):

    # Recursion protection
    if isinstance (samix_fcisolver, StateAverageNMixFCISolver):
        return samix_fcisolver

    class FCISolver (samix_fcisolver.__class__, StateAverageNMixFCISolver):

        def _get_nelec (self, solver, nelec):
            m = solver.spin if solver.spin is not None else 0
            c = getattr (solver, 'charge', 0) or 0
            if m or c:
                nelec = np.sum (nelec) - c
                nelec = (nelec+m)//2, (nelec-m)//2
            return nelec

    sanmix_fcisolver = FCISolver (samix_fcisolver.mol)
    sanmix_fcisolver.__dict__.update (samix_fcisolver.__dict__)
    return sanmix_fcisolver

def state_average_n_mix (casscf, fcisolvers, weights=(0.5,0.5)):
    sacasscf = state_average_mix (casscf, fcisolvers, weights=weights)
    sacasscf.fcisolver = get_sanmix_fcisolver (sacasscf.fcisolver)
    return sacasscf

def state_average_n_mix_(casscf, fcisolvers, weights=(0.5,0.5)):
    sacasscf = state_average_n_mix (casscf, fcisolvers, weights)
    casscf.__class__ = sacasscf.__class__
    casscf.__dict__.update(sacasscf.__dict__)
    return casscf

class H1EZipFCISolver (object):
    pass

def get_h1e_zipped_fcisolver (fcisolver):
    ''' Wrap a state-average-mix FCI solver to take a list of h1es to apply to each state.
    I'm not sure how orthogonality works into this, but in the most straightforward
    application of SA-LASSCF all the product states will have different local symmetries
    so maybe it doesn't matter. (Doing the same for orbital number would be a gigantic pain
    in the ass because of the rdms.)'''

    # Recursion protection
    if isinstance (fcisolver, H1EZipFCISolver):
        return fcisolver

    assert isinstance (fcisolver, StateAverageMixFCISolver), 'requires StateAverageMixFCISolver'

    class FCISolver (fcisolver.__class__, H1EZipFCISolver):

        def kernel(self, h1, h2, norb, nelec, ci0=None, verbose=0, ecore=0, orbsym=None, **kwargs):
            # Note self.orbsym is initialized lazily in mc1step_symm.kernel function
            log = logger.new_logger(self, verbose)
            es = []
            cs = []
            if isinstance (ecore, (int, float, np.integer, np.floating)):
                ecore = [ecore,] * len (h1)
            if orbsym is None: orbsym=self.orbsym
            for solver, my_args, my_kwargs in self._loop_solver(_state_arg (ci0), _state_arg (h1), _state_arg (ecore)):
                c0 = my_args[0]
                h1e = my_args[1]
                e0 = my_args[2]
                e, c = solver.kernel(h1e, h2, norb, self._get_nelec(solver, nelec), c0,
                                     orbsym=orbsym, verbose=log, ecore=e0, **kwargs)
                if solver.nroots == 1:
                    es.append(e)
                    cs.append(c)
                else:
                    es.extend(e)
                    cs.extend(c)
            self.e_states = es
            self.converged = np.all(getattr(sol, 'converged', True)
                                       for sol in self.fcisolvers)

            if log.verbose >= logger.DEBUG:
                if has_spin_square:
                    ss = self.states_spin_square(cs, norb, nelec)[0]
                    for i, ei in enumerate(es):
                        log.debug('state %d  E = %.15g S^2 = %.7f', i, ei, ss[i])
                else:
                    for i, ei in enumerate(es):
                        log.debug('state %d  E = %.15g', i, ei)
            return np.einsum('i,i', np.array(es), self.weights), cs

        def approx_kernel(self, h1, h2, norb, nelec, ci0=None, orbsym=None, **kwargs):
            es = []
            cs = []
            if orbsym is None: orbsym=self.orbsym
            for solver, my_args, _ in self._loop_solver (_state_arg (ci0), _state_arg (h1)):
                c0 = my_args[0]
                h1e = my_args[1]
                try:
                    e, c = solver.approx_kernel(h1e, h2, norb, self._get_nelec(solver, nelec), c0,
                                                orbsym=orbsym, **kwargs)
                except AttributeError:
                    e, c = solver.kernel(h1e, h2, norb, self._get_nelec(solver, nelec), c0,
                                         orbsym=orbsym, **kwargs)
                if solver.nroots == 1:
                    es.append(e)
                    cs.append(c)
                else:
                    es.extend(e)
                    cs.extend(c)
            return np.einsum('i,i->', es, weights), cs

        def states_absorb_h1e (self, h1, h2, norb, nelec, fac):
            op = []
            for solver, my_args, _ in self._loop_solver (_state_arg (h1)):
                h1e = my_args[0]
                op.append (solver.absorb_h1e (h1e, h2, norb, self._get_nelec (solver, nelec), fac) if h1 is not None else h2)
            return op

        def states_contract_2e (self, h2, ci, norb, nelec, link_index=None):
            hc = []
            for solver, my_args, _ in self._loop_solver (_state_arg (ci), _state_arg (h2), _solver_arg (link_index)):
                c0 = my_args[0]
                h2e = my_args[1]
                linkstr = my_args[2]
                hc.append (solver.contract_2e (h2e, c0, norb, self._get_nelec (solver, nelec), link_index=linkstr))
            return hc

        def states_make_hdiag (self, h1, h2, norb, nelec):
            hdiag = []
            for solver, my_args, _ in self._loop_solver (_state_arg (h1)):
                h1e = my_args[0]
                hdiag.append (solver.make_hdiag (h1e, h2, norb, self._get_nelec (solver, nelec)))
            return hdiag

        def states_make_hdiag_csf (self, h1, h2, norb, nelec):
            hdiag = []
            for solver, my_args, _ in self._loop_solver (_state_arg (h1)):
                h1e = my_args[0]
                with temporary_env (solver, orbsym=self.orbsym):
                    hdiag.append (solver.make_hdiag_csf (h1e, h2, norb, self._get_nelec (solver, nelec)))
            return hdiag

        # The below can conceivably be added to pyscf.mcscf.addons.StateAverageMixFCISolver in future

        def states_gen_linkstr (self, norb, nelec, tril=True):
            linkstr = []
            for solver in self.fcisolvers:
                with temporary_env (solver, orbsym=self.orbsym):
                    linkstr.append (solver.gen_linkstr (norb, self._get_nelec (solver, nelec), tril=tril)
                        if getattr (solver, 'gen_linkstr', None) else None)
            return linkstr
                    
        def states_transform_ci_for_orbital_rotation (self, ci0, norb, nelec, umat):
            ci1 = []
            for solver, my_args, _ in self._loop_solver (_state_arg (ci0)):
                ne = self._get_nelec (solver, nelec)
                ci0_i = my_args[0].reshape ([special.comb (norb, n, exact=True) for n in ne])
                ci1.append (solver.transform_ci_for_orbital_rotation (ci0_i, norb, ne, umat))
            return ci1

        def states_trans_rdm12s (self, ci1, ci0, norb, nelec, link_index=None, **kwargs):
            ci1 = _state_arg (ci1)
            ci0 = _state_arg (ci0)
            link_index = _solver_arg (link_index)
            nelec = _solver_arg ([self._get_nelec (solver, nelec) for solver in self.fcisolvers])
            tdm1 = []
            tdm2 = []
            for dm1, dm2 in self._collect ('trans_rdm12s', ci1, ci0, norb, nelec, link_index=link_index, **kwargs):
                tdm1.append (dm1)
                tdm2.append (dm2)
            return tdm1, tdm2

        # DANGER! DANGER WILL ROBINSON! I KNOW THAT THE BELOW MAY MAKE SOME THINGS CONVENIENT BUT THERE COULD BE MANY UNFORSEEN PROBLEMS!
        absorb_h1e = states_absorb_h1e
        contract_2e = states_contract_2e
        make_hdiag = states_make_hdiag

    h1ezipped_fcisolver = FCISolver (fcisolver.mol)
    h1ezipped_fcisolver.__dict__.update (fcisolver.__dict__)
    return h1ezipped_fcisolver

