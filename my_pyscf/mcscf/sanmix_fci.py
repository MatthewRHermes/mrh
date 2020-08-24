import numpy as np
from pyscf.mcscf.addons import StateAverageMCSCFSolver, StateAverageMixFCISolver, state_average_mix

class StateAverageNMixFciSolver (StateAverageMixFCISolver):
    pass

def get_sanmix_fcisolver (samix_fcisolver):

    class FCISolver (samix_fcisolver.__class__, StateAverageNMixFciSolver):

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


