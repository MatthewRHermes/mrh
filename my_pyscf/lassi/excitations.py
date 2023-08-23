import numpy as np
from scipy import linalg
from pyscf.fci.direct_spin1 import _unpack_nelec
from mrh.my_pyscf.mcscf.productstate import ProductStateFCISolver, ImpureProductStateFCISolver, state_average_fcisolver
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.lassi import op_o0, op_o1
from pyscf.lib import temporary_env
op = (op_o0, op_o1)

class ExcitationPSFCISolver (ProductStateFCISolver):
    def __init__(self, solver_ref, ci_ref, norb_ref, nelec_ref, orbsym_ref=None,
                 wfnsym_ref=None, stdout=None, verbose=0, opt=0, **kwargs):
        self.solver_ref = solver_ref
        self.ci_ref = ci_ref
        self.norb_ref = np.asarray (norb_ref)
        self.nelec_ref = nelec_ref
        self.orbsym_ref = orbsym_ref
        self.wfnsym_ref = wfnsym_ref
        self.active_frags = None
        self.active_orb_idx = None
        self.active_orbs = None
        self.opt = opt
        ProductStateFCISolver.__init__(self, solver_ref.fcisolvers, stdout=stdout,
                                       verbose=verbose)
        self.dm1s_ref = self.solver_ref.make_rdm1s (self.ci_ref, norb_ref, nelec_ref)
        self.dm2_ref = self.solver_ref.make_rdm2 (self.ci_ref, norb_ref, nelec_ref)

    def set_active_fragments_(self, active_frags):
        self.active_frags = active_frags
        nj = np.cumsum (self.norb_ref)
        ni = nj - self.norb_ref
        idx = np.zeros (len (self.norb_ref), dtype=bool)
        for ifrag in active_frag:
            i, j = ni[ifrag], nj[ifrag]
            idx[i:j] = True
        self.active_orb_idx = idx
        self.active_orbs = np.where (idx)[0]

    def get_dm_spectator (self):
        dm1s = self.dm1s_ref.copy ()
        dm2 = self.dm2_ref.copy ()
        idx = self.active_orb_idx
        dm1s[:,idx,:] = 0
        dm1s[:,:,idx] = 0
        dm2[idx,:,:,:] = 0
        dm2[:,idx,:,:] = 0
        dm2[:,:,idx,:] = 0
        dm2[:,:,:,idx] = 0
        return dm1s, dm2

    def diag_ham_qq (self, h1, h2):
        ci_fr = [[c] for c in self.ci_ref]
        nelec_frs = np.asarray ([[list(self.solver_ref._get_nelec (s, n))
                                  for s, n in zip (self.solver_ref.fcisolvers, self.nelec_ref)]])
        with temporary_env (self, ncas_sub=self.norb_ref):
            ham_qq = op[self.opt].ham (self, h1, h2, ci_fr, nelec_frs, soc=0,
                                       orbsym=self.orbsym_ref, wfnsym=self.wfnsym_ref)
        return linalg.eigh (ham_qq)

    def set_excitation_character_(self, delta_neleca, delta_nelecb, delta_smult, lweights=None):
        # TODO: point group symmetry
        delta = np.stack ([delta_neleca, delta_nelecb, delta_smult], axis=1)
        idx_active = np.any (delta!=0, axis=1)
        active_frags = np.where (idx_active)[0]
        self.set_active_fragments_(active_frags)
        fcisolvers = []
        delta_charge = delta_neleca + delta_nelecb
        delta_spin = delta_neleca - delta_nelecb
        for ifrag in active_frags:
            s_ref = self.solver_ref.fcisolvers[ifrag_ref]
            mol = s_ref.mol
            nelec_ref = _unpack_nelec (self.nelec_ref[ifrag])
            spin_ref = nelec_ref[0] - nelec_ref[1]
            charge = getattr (s_ref, 'charge', 0) + delta_charge[ifrag]
            spin = getattr (s_ref, 'spin', spin_ref) + delta_spin[ifrag]
            smult = getattr (s_ref, 'smult', abs(spin_ref)+1) + delta_smult[ifrag]
            fcisolvers.append (csf_solver (mol, smult=smult).set (charge=charge, spin=spin))
        if lweights is not None:
            for ix, (solver, weights) in enumerate (zip (fcisolvers, lweights)):
                if len (weights) > 1:
                    fcisolvers[ix] = state_average_fcisolver (solver, weights=weights)
        self.fcisolvers = fcisolvers

    def kernel (self, h1, h2, norb_f, nelec_f, ecore=0, ci0=None, orbsym=None,
                conv_tol_grad=1e-4, conv_tol_self=1e-10, max_cycle_macro=50,
                **kwargs):
        dm1s_ref, dm2_ref = self.get_dm_spectator ()
        h1eff, h0eff = self.project_hfrag (h1, h2, self.ci_ref, norb_f, nelec_f, ecore=ecore,
                                           dm1s=dm1s_ref, dm2=dm2_ref)
        idx = self.active_orb_idx
        h2 = h[idx][:,idx][:,:,idx][:,:,:,idx]
        h1 = linalg.block_diag (*[h1eff[ifrag] for ifrag in self.active_frags])
        h0 = np.mean (h0eff) + ecore
        norb_f = np.asarray ([norb_f[ifrag] for ifrag in self.active_frags])
        nelec_f = np.asarray ([nelec_f[ifrag] for ifrag in self.active_frags])
        if orbsym is not None:
            orbsym = [orbsym[iorb] if iorb in self.active_orbs]
        converged, energy_elec, ci1_active = ProductStateFCISolver.kernel (
            self, h1, h2, norb_f, nelec_f, ecore=h0, ci0=ci0, orbsym=orbsym,
            conv_tol_grad=conv_tol_grad, conv_tol_self=conv_tol_self,
            max_cycle_macro=max_cycle_macro, **kwargs
        )
        ci1 = [c for c in self.ci_ref]
        for ifrag, c in zip (self.active_frags, ci1_active):
            ci1[ifrag] = c
        return converged, energy_elec, ci1_active


