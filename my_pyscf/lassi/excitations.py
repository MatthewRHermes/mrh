import numpy as np
from scipy import linalg
from pyscf.fci.direct_spin1 import _unpack_nelec
from mrh.my_pyscf.mcscf.productstate import ProductStateFCISolver, ImpureProductStateFCISolver, state_average_fcisolver
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.lassi import op_o0, op_o1
from mrh.my_pyscf.lassi.citools import get_lroots
from pyscf import lib
from pyscf.lib import temporary_env
op = (op_o0, op_o1)

def only_ground_states (ci0):
    # TODO: examine whether this should be considered a limitation
    ci1 = []
    for c in ci0:
        if c.ndim==3: c = c[0]
        ci1.append (c)
    return ci1

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

    def kernel (self, h1, h2, ecore=0,
                conv_tol_grad=1e-4, conv_tol_self=1e-10, max_cycle_macro=50,
                **kwargs):
        h0, h1, h2 = self.get_active_h (ecore, h1, h2)
        norb_f = np.asarray ([self.norb_ref[ifrag] for ifrag in self.active_frags])
        nelec_f = np.asarray ([self.nelec_ref[ifrag] for ifrag in self.active_frags])
        orbsym = self.orbsym_ref
        if orbsym is not None:
            idx = self.get_active_orb_idx ()
            orbsym = [orbsym[iorb] for iorb in range (norb_tot) if idx[iorb]]
        # TODO: point group symmetry; I probably also have to do something to wfnsym
        ci0 = self.get_init_guess (h1, h2)
        converged, energy_elec, ci1_active = ProductStateFCISolver.kernel (
            self, h1, h2, norb_f, nelec_f, ecore=h0, ci0=ci0, orbsym=orbsym,
            conv_tol_grad=conv_tol_grad, conv_tol_self=conv_tol_self,
            max_cycle_macro=max_cycle_macro, **kwargs
        )
        ci1 = [c for c in self.ci_ref]
        for ifrag, c in zip (self.active_frags, ci1_active):
            ci1[ifrag] = np.asarray (c)
        return converged, energy_elec, ci1

    def get_ham_pq (self, h1, h2, ci_p):
        active_frags = [ifrag for ifrag in self.active_frags]
        fcisolvers, nelec_ref = self.solver_ref.fcisolvers, self.nelec_ref
        fcisolvers = [fcisolvers[ifrag] for ifrag in active_frags]
        nelec_ref = [nelec_ref[ifrag] for ifrag in active_frags]
        norb_ref = [self.norb_ref[ifrag] for ifrag in active_frags]
        ci_q = [self.ci_ref[ifrag] for ifrag in active_frags]
        ci_fr = [[cp, cq] for cp, cq in zip (ci_p, ci_q)]
        nelec_fs_q = np.asarray ([list(self._get_nelec (s, n)) 
                                   for s, n in zip (fcisolvers, nelec_ref)])
        fcisolvers = self.fcisolvers
        nelec_fs_p = np.asarray ([list(self._get_nelec (s, n)) 
                                   for s, n in zip (fcisolvers, nelec_ref)])
        nelec_frs = np.stack ([nelec_fs_p, nelec_fs_q], axis=1)
        with temporary_env (self, ncas_sub=norb_ref, mol=fcisolvers[0].mol):
            ham_pq = op[self.opt].ham (self, h1, h2, ci_fr, nelec_frs, soc=0,
                                       orbsym=self.orbsym_ref, wfnsym=self.wfnsym_ref)[0]
        return ham_pq

    def op_ham_pq_ref (self, h1, h2, ci):
        # TODO: point group symmetry
        active_frags = [ifrag for ifrag in self.active_frags]
        norb_f = [self.norb_ref[ifrag] for ifrag in active_frags]
        nelec_f = np.asarray ([self.nelec_ref[ifrag] for ifrag in active_frags])
        ref_fcisolvers = [self.solver_ref.fcisolvers[ifrag] for ifrag in active_frags]
        ci_fr_ket = [[self.ci_ref[ifrag]] for ifrag in active_frags]
        nelec_rfs_ket = np.asarray ([[list(self._get_nelec (s, n))
                                     for s, n in zip (ref_fcisolvers, nelec_f)]])
        nelec_frs_ket = nelec_rfs_ket.transpose (1,0,2)
        ci_fr_bra = [[c] for c in only_ground_states (ci)]
        nelec_rfs_bra = np.asarray ([[list(self._get_nelec (s, n))
                                     for s, n in zip (self.fcisolvers, nelec_f)]])
        nelec_frs_bra = nelec_rfs_bra.transpose (1,0,2)
        h_op = op[self.opt].contract_ham_ci
        with temporary_env (self, ncas_sub=norb_f, mol=self.fcisolvers[0].mol):
            hci_fr_pabq = h_op (self, h1, h2, ci_fr_ket, nelec_frs_ket, ci_fr_bra, nelec_frs_bra,
                                soc=0, orbsym=None, wfnsym=None)
        hci_f_abq = [hc[0] for hc in hci_fr_pabq]
        return hci_f_abq

    def get_init_guess (self, h1, h2):
        norb_f = np.asarray ([self.norb_ref[ifrag] for ifrag in self.active_frags])
        nelec_f = np.asarray ([self.nelec_ref[ifrag] for ifrag in self.active_frags])
        ci0 = self._check_init_guess (None, norb_f, nelec_f, h1, h2)
        # Find lowest-energy ENV, including VRV contributions
        lroots = get_lroots (ci0)
        ham_pq = self.get_ham_pq (h1, h2, ci0)
        p = np.prod (lroots)
        h_pp = ham_pq[:p,:p]
        h_pq = ham_pq[:p,p:]
        h_qq = ham_pq[p:,p:]
        q = ham_pq.shape[-1] - p
        e0, si = linalg.eigh (ham_pq)
        e_q, si_q = linalg.eigh (h_qq)
        h_pq = np.dot (h_pq, si_q)
        denom = e0[0] - e_q
        idx = np.abs (denom) > 1e-16
        heff_pp = h_pp + np.dot (h_pq[:,idx].conj () / denom[None,idx], h_pq[:,idx].T)
        e_p = np.diag (heff_pp)
        idxmin = np.argmin (e_p)
        # ENV index to address
        idx, pj = idxmin, p
        addr = []
        for ifrag, lroot in enumerate (lroots[:-1]):
            pj = pj // lroot
            idx, j = divmod (idx, pj)
            addr.append (j)
        addr.append (idx)
        # Sort against this reference state
        nfrag = len (addr)
        e_p_arr = e_p.reshape (*lroots[::-1]).T
        sort_idxs = []
        h_pp = h_pp.reshape (*(list(lroots[::-1])*2))
        h_pq = ham_pq[:p,p:].reshape (*(list(lroots[::-1])+[q,]))
        for ifrag in range (nfrag):
            if lroots[ifrag]<2: continue
            e_p_slice = e_p_arr
            for jfrag in range (ifrag):
                e_p_slice = e_p_slice[addr[jfrag]]
            for jfrag in range (ifrag+1,nfrag):
                e_p_slice = e_p_slice[:,addr[jfrag]]
            sort_idx = np.argsort (e_p_slice)
            assert (sort_idx[0] == addr[ifrag])
            ci0[ifrag] = np.stack ([ci0[ifrag][i] for i in sort_idx], axis=0)
            dimorder = list(range(h_pp.ndim))
            dimorder.insert (0, dimorder.pop (nfrag-(1+ifrag)))
            dimorder.insert (1, dimorder.pop (2*nfrag-(1+ifrag)))
            h_pp = h_pp.transpose (*dimorder)
            h_pp = h_pp[sort_idx,...][:,sort_idx,...]
            dimorder = np.argsort (dimorder)
            h_pp = h_pp.transpose (*dimorder)
            dimorder = list(range(h_pq.ndim))
            dimorder.insert (0, dimorder.pop (nfrag-(1+ifrag)))
            h_pq = h_pq.transpose (*dimorder)
            h_pq = h_pq[sort_idx,...]
            dimorder = np.argsort (dimorder)
            h_pq = h_pq.transpose (*dimorder)
        h_pp = h_pp.reshape (p, p)
        h_pq = h_pq.reshape (p, q)
        #ham_pq_test = np.zeros_like (ham_pq)
        #ham_pq_test[:p,:p] = h_pp
        #ham_pq_test[:p,p:] = h_pq
        #ham_pq_test[p:,:p] = h_pq.T
        #ham_pq_test[p:,p:] = h_qq
        #ham_pq_ref = self.get_ham_pq (h1, h2, ci0)
        #assert (abs (lib.fp (ham_pq_test)-lib.fp (ham_pq_ref)) < 1e-8)
        #e0_test = linalg.eigh (ham_pq_test)[0]
        #assert (abs (lib.fp (e0_test)-lib.fp (e0)) < 1e-8)

        # Something like the below?
        #ham_pq[:p,:p] = h_pp
        #ham_pq[:p,p:] = h_pq
        #ham_pq[p:,:p] = h_pq.T
        #idx = np.ones (len (ham_pq), dtype=bool)
        #if p>1: idx[1:p] = False
        #ham_pq = ham_pq[idx,:][:,idx]
        #e0 = linalg.eigh (ham_pq)[0][0]
        #print (ham_pq[0,0] - e0, e0)
        ci0 = self._check_init_guess (ci0, norb_f, nelec_f, h1, h2)
        return ci0




