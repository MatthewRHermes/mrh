import numpy as np
from pyscf.lib import logger
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

        def kernel(self, h1, h2, norb, nelec, ci0=None, verbose=0, ecore=0, **kwargs):
            # Note self.orbsym is initialized lazily in mc1step_symm.kernel function
            log = logger.new_logger(self, verbose)
            es = []
            cs = []
            if isinstance (ecore, (np.integer, np.floating)):
                ecore = [ecore,] * len (h1)
            for solver, my_args, my_kwargs in self._loop_solver(_state_arg (ci0), _state_arg (h1), _state_arg (ecore)):
                c0 = my_args[0]
                h1e = my_args[1]
                e0 = my_args[2]
                e, c = solver.kernel(h1e, h2, norb, self._get_nelec(solver, nelec), c0,
                                     orbsym=self.orbsym, verbose=log, ecore=e0, **kwargs)
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

        def approx_kernel(self, h1, h2, norb, nelec, ci0=None, **kwargs):
            es = []
            cs = []
            for ix, (solver, my_args, my_kwargs) in enumerate (self._loop_solver (_state_arg (ci0), _state_arg (h1))):
                c0 = my_args[0]
                h1e = my_args[1]
                try:
                    e, c = solver.approx_kernel(h1e, h2, norb, self._get_nelec(solver, nelec), c0,
                                                orbsym=self.orbsym, **kwargs)
                except AttributeError:
                    e, c = solver.kernel(h1e, h2, norb, self._get_nelec(solver, nelec), c0,
                                         orbsym=self.orbsym, **kwargs)
                if solver.nroots == 1:
                    es.append(e)
                    cs.append(c)
                else:
                    es.extend(e)
                    cs.extend(c)
            return np.einsum('i,i->', es, weights), cs

        def states_absorb_h1e (self, h1, h2, norb, nelec, fac):
            op = []
            for ix, (solver, my_args, my_kwargs) in enumerate (self._loop_solver (_state_arg (h1))):
                h1e = my_args[0]
                op.append (solver.absorb_h1e (h1e, h2, norb, self._get_nelec (solver, nelec), fac) if h1 is not None else h2)
            return op

        def states_contract_2e (self, h2, ci, norb, nelec, link_index=None):
            hc = []
            for ix, (solver, my_args, my_kwargs) in enumerate (self._loop_solver (_state_arg (ci), _state_arg (h2), _solver_arg (link_index))):
                c0 = my_args[0]
                h2e = my_args[1]
                linkstr = my_args[2]
                hc.append (solver.contract_2e (h2e, c0, norb, self._get_nelec (solver, nelec), link_index=linkstr))
            return hc

        def states_make_hdiag (self, h1, h2, norb, nelec):
            hdiag = []
            for ix, (solver, my_args, my_kwargs) in enumerate (self._loop_solver (_state_arg (h1))):
                h1e = my_args[0]
                hdiag.append (solver.make_hdiag (h1e, h2, norb, self._get_nelec (solver, nelec)))
            return hdiag

        def states_gen_linkstr (self, norb, nelec, tril=True):
            return [solver.gen_linkstr (norb, self._get_nelec (solver, nelec), tril=tril)
                if getattr (solver, 'gen_linkstr', None) else None
                for solver in self.fcisolvers]
                    

        # DANGER! DANGER WILL ROBINSON! I KNOW THAT THE BELOW MAY MAKE SOME THINGS CONVENIENT BUT THERE COULD BE MANY UNFORSEEN PROBLEMS!
        absorb_h1e = states_absorb_h1e
        contract_2e = states_contract_2e
        make_hdiag = states_make_hdiag

    h1ezipped_fcisolver = FCISolver (fcisolver.mol)
    h1ezipped_fcisolver.__dict__.update (fcisolver.__dict__)
    return h1ezipped_fcisolver

