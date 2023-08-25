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
        self.opt = opt
        ProductStateFCISolver.__init__(self, solver_ref.fcisolvers, stdout=stdout,
                                       verbose=verbose)
        self.dm1s_ref = np.asarray (self.solver_ref.make_rdm1s (self.ci_ref, norb_ref, nelec_ref))
        self.dm2_ref = self.solver_ref.make_rdm2 (self.ci_ref, norb_ref, nelec_ref)
        self.active_frags = {}

    @property
    def fcisolvers (self):
        return [solver for ifrag, solver in self.active_frags.items ()]
    @fcisolvers.setter
    def fcisolvers (self, solvers):
        self.active_frags = {i: s for i, s in enumerate (solvers)}

    def get_active_orb_idx (self):
        nj = np.cumsum (self.norb_ref)
        ni = nj - self.norb_ref
        idx = np.zeros (nj[-1], dtype=bool)
        for ifrag, solver in self.active_frags.items ():
            i, j = ni[ifrag], nj[ifrag]
            idx[i:j] = True
        return idx

    def get_active_h (self, h0, h1, h2):
        idx = self.get_active_orb_idx ()
        dm1s = self.dm1s_ref.copy ()
        dm2 = self.dm2_ref.copy ()
        dm1s[:,idx,:] = 0
        dm1s[:,:,idx] = 0
        dm2[idx,:,:,:] = 0
        dm2[:,idx,:,:] = 0
        dm2[:,:,idx,:] = 0
        dm2[:,:,:,idx] = 0
        h1eff, h0eff = self.project_hfrag (h1, h2, self.ci_ref, self.norb_ref, self.nelec_ref, 
                                           ecore=h0, dm1s=dm1s, dm2=dm2)
        h1eff = [h1eff[ifrag] for ifrag in self.active_frags]
        h0eff = [h0eff[ifrag] for ifrag in self.active_frags]
        h2 = h2[idx][:,idx][:,:,idx][:,:,:,idx]
        h1a = linalg.block_diag (*[h[0] for h in h1eff])
        h1b = linalg.block_diag (*[h[1] for h in h1eff])
        h1 = np.stack ([h1a, h1b], axis=0)
        h0 = np.mean (h0eff)
        return h0, h1, h2

    def diag_ham_qq (self, h1, h2):
        ci_fr = [[c] for c in self.ci_ref]
        nelec_frs = np.asarray ([[list(self._get_nelec (s, n))
                                  for s, n in zip (self.solver_ref.fcisolvers, self.nelec_ref)]])
        with temporary_env (self, ncas_sub=self.norb_ref):
            ham_qq = op[self.opt].ham (self, h1, h2, ci_fr, nelec_frs, soc=0,
                                       orbsym=self.orbsym_ref, wfnsym=self.wfnsym_ref)
        return linalg.eigh (ham_qq)

    def set_excited_fragment_(self, ifrag, delta_neleca, delta_nelecb, delta_smult, weights=None):
        # TODO: point group symmetry
        delta_charge = -(delta_neleca + delta_nelecb)
        delta_spin = delta_neleca - delta_nelecb
        s_ref = self.solver_ref.fcisolvers[ifrag]
        mol = s_ref.mol
        nelec_ref = _unpack_nelec (self.nelec_ref[ifrag])
        spin_ref = nelec_ref[0] - nelec_ref[1]
        charge = getattr (s_ref, 'charge', 0) + delta_charge
        spin = getattr (s_ref, 'spin', spin_ref) + delta_spin
        smult = getattr (s_ref, 'smult', abs(spin_ref)+1) + delta_smult
        nelec_a = nelec_ref[0] + (delta_spin - delta_charge) // 2 
        nelec_b = nelec_ref[1] - (delta_spin + delta_charge) // 2 
        nelec = tuple ((nelec_a, nelec_b))
        fcisolver = csf_solver (mol, smult=smult).set (charge=charge, spin=spin,
                                                       nelec=nelec, norb=s_ref.norb)
        if hasattr (weights, '__len__') and len (weights) > 1:
            fcisolver = state_average_fcisolver (fcisolver, weights=weights)
        self.active_frags[ifrag] = fcisolver

    def kernel (self, h1, h2, norb_f, nelec_f, ecore=0,
                conv_tol_grad=1e-4, conv_tol_self=1e-10, max_cycle_macro=50,
                **kwargs):
        h0, h1, h2 = self.get_active_h (ecore, h1, h2)
        norb_f = np.asarray ([norb_f[ifrag] for ifrag in self.active_frags])
        nelec_f = np.asarray ([nelec_f[ifrag] for ifrag in self.active_frags])
        orbsym = self.orbsym_ref
        if orbsym is not None:
            idx = self.get_active_orb_idx ()
            orbsym = [orbsym[iorb] for iorb in range (norb_tot) if idx[iorb]]
        converged, energy_elec, ci1_active = ProductStateFCISolver.kernel (
            self, h1, h2, norb_f, nelec_f, ecore=h0, ci0=None, orbsym=orbsym,
            conv_tol_grad=conv_tol_grad, conv_tol_self=conv_tol_self,
            max_cycle_macro=max_cycle_macro, **kwargs
        )
        ci1 = [c for c in self.ci_ref]
        for ifrag, c in zip (self.active_frags, ci1_active):
            ci1[ifrag] = np.asarray (c)
        return converged, energy_elec, ci1


