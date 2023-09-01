import copy
import numpy as np
from scipy import linalg
from pyscf.fci.direct_spin1 import _unpack_nelec
from pyscf.mcscf.addons import StateAverageFCISolver
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
        c = np.asarray (c)
        if c.ndim==3: c = c[0]
        ci1.append (c)
    return ci1

class _vrvloop_env (object):
    def __init__(self, fciobj, vrvsolvers, e_q, si_q):
        self.fciobj = fciobj
        self.vrvsolvers = vrvsolvers
        self.e_q = e_q
        self.si_q = si_q
    def __enter__(self):
        self.fciobj.fcisolvers = self.vrvsolvers
        self.fciobj._e_q = self.e_q
        self.fciobj._si_q = self.si_q
    def __exit__(self, type, value, traceback):
        self.fciobj.revert_vrvsolvers_()

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
        self._deactivate_vrv = False # for testing
        ProductStateFCISolver.__init__(self, solver_ref.fcisolvers, stdout=stdout,
                                       verbose=verbose)
        self.dm1s_ref = np.asarray (self.solver_ref.make_rdm1s (self.ci_ref, norb_ref, nelec_ref))
        self.dm2_ref = self.solver_ref.make_rdm2 (self.ci_ref, norb_ref, nelec_ref)
        self.active_frags = []
        self.fcisolvers = []
        self._e_q = []
        self._si_q = []

    def get_active_orb_idx (self):
        nj = np.cumsum (self.norb_ref)
        ni = nj - self.norb_ref
        idx = np.zeros (nj[-1], dtype=bool)
        for ifrag, solver in zip (self.active_frags, self.fcisolvers):
            i, j = ni[ifrag], nj[ifrag]
            idx[i:j] = True
        return idx

    def get_active_h (self, h0, h1, h2):
        idx = self.get_active_orb_idx ()
        dm1s = self.dm1s_ref.copy ()
        dm2 = self.dm2_ref.copy ()
        norb_active = np.count_nonzero (idx)
        idx = list (np.where (idx)[0]) + list (np.where (~idx)[0])
        dm1s = dm1s[:,idx][:,:,idx]
        h1 = h1[idx][:,idx]
        dm2 = dm2[idx][:,idx][:,:,idx][:,:,:,idx]
        h2 = h2[idx][:,idx][:,:,idx][:,:,:,idx]
        norb_ref = [norb_active,] + [n for ifrag, n in enumerate (self.norb_ref)
                                     if not (ifrag in self.active_frags)]
        h1eff, h0eff = self.project_hfrag (h1, h2, self.ci_ref, norb_ref, self.nelec_ref, 
                                           ecore=h0, dm1s=dm1s, dm2=dm2)
        h0, h1 = h0eff[0], h1eff[0]
        h2 = h2[:norb_active][:,:norb_active][:,:,:norb_active][:,:,:,:norb_active]
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
        if ifrag in self.active_frags:
            self.fcisolvers[self.active_frags.index (ifrag)] = fcisolver
        else:
            self.active_frags.append (ifrag)
            self.fcisolvers.append (fcisolver)
            idx = np.argsort (self.active_frags)
            self.active_frags = [self.active_frags[i] for i in idx]
            self.fcisolvers = [self.fcisolvers[i] for i in idx]

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
        ci0, vrvsolvers, e_q, si_q = self.prepare_vrvsolvers_(h0, h1, h2)
        with _vrvloop_env (self, vrvsolvers, e_q, si_q):
            converged, energy_elec, ci1_active = ProductStateFCISolver.kernel (
                self, h1, h2, norb_f, nelec_f, ecore=h0, ci0=ci0, orbsym=orbsym,
                conv_tol_grad=conv_tol_grad, conv_tol_self=conv_tol_self,
                max_cycle_macro=max_cycle_macro, **kwargs
            )
        #hci_f_abq = self.op_ham_pq_ref (h1, h2, ci1_active)
        #h1eff, h0eff = self.project_hfrag (h1, h2, ci1_active, norb_f, nelec_f,
        #                                   ecore=h0)#, **kwargs)
        #e_vrv, ci1_vrv = self._1shot_with_vrv (
        #    h0eff, h1eff, h2, hci_f_abq, h_qq,
        #    ci1_active, norb_f, nelec_f, orbsym=orbsym, 
        #    **kwargs
        #)  
        #for ix, (c_active, c_vrv) in enumerate (zip (ci1_active, ci1_vrv)):
        #    c_active = np.asarray (c_active)
        #    ndeta, ndetb = c_active.shape[-2:]
        #    c_active = c_active.reshape (-1, ndeta*ndetb)
        #    c_vrv = np.asarray (c_vrv).reshape (-1, ndeta*ndetb)
        #    ovlp = np.dot (c_active.conj (), c_vrv.T)
        ci1 = [c for c in self.ci_ref]
        for ifrag, c in zip (self.active_frags, ci1_active):
            ci1[ifrag] = np.asarray (c)
        return converged, energy_elec, ci1

    def project_hfrag (self, h1, h2, ci, norb_f, nelec_f, ecore=0, dm1s=None, dm2=None, **kwargs):
        h1eff, h0eff = ProductStateFCISolver.project_hfrag (
            self, h1, h2, ci, norb_f, nelec_f, ecore=ecore, dm1s=dm1s, dm2=dm2, **kwargs
        )
        # hijack this function to set the perturber vectors within the productstate kernel loop
        if len (self._e_q) and not self._deactivate_vrv:
            hci_f_abq = self.op_ham_pq_ref (h1, h2, ci)
            for hci_abq, solver in zip (hci_f_abq, self.fcisolvers):
                solver.vrv_qab = np.tensordot (self._si_q, hci_abq, axes=((0),(-1)))
        return h1eff, h0eff

    def energy_elec (self, h1, h2, ci, norb_f, nelec_f, ecore=0, **kwargs):
        energy_tot = ProductStateFCISolver.energy_elec (
            self, h1, h2, ci, norb_f, nelec_f, ecore=ecore, **kwargs
        )
        # Also compute the vrv perturbation energy
        if len (self._e_q) and not self._deactivate_vrv:
            ci0 = []
            # extract this from the putatively converged solver cycles
            # if you attempt to recalculate it, then it has to be reconverged i guess
            denom_q = 0
            for c, solver in zip (ci, self.fcisolvers):
                t = solver.transformer
                c = np.asarray (c).reshape (-1, t.ndeta, t.ndetb)
                ci0.append (c)
                denom_q += solver.denom_q
            denom_q /= len (ci)
            lroots = get_lroots (ci0)
            p = np.prod (lroots)
            ham_pq = self.get_ham_pq (ecore, h1, h2, ci0)
            idx = np.ones (len (ham_pq), dtype=bool)
            idx[1:p] = False
            e0, si = linalg.eigh (ham_pq[idx,:][:,idx])
            #idx = (si[0].conj () * si[0]) > 1e-16
            #e0, si = e0[idx], si[:,idx]
            h_pp = ham_pq[:p,:p]
            h_pq = ham_pq[:p,p:]
            h_qq = ham_pq[p:,p:]
            h_qq = self._si_q.conj ().T @ h_qq @ self._si_q
            h_pq = np.dot (h_pq, self._si_q)
            ham_pq = ham_pq[idx,:][:,idx]
            idx = np.abs (denom_q) > 1e-16
            e_p = np.diag (np.dot (h_pq[:,idx].conj () / denom_q[None,idx], h_pq[:,idx].T))
            e_p = e_p.reshape (*lroots[::-1]).T
            for solver in self.fcisolvers:
                if hasattr (getattr (solver, 'weights', None), '__len__'):
                    e_p = np.dot (solver.weights, e_p)
                else:
                    e_p = e_p[0]
            energy_tot += e_p
        return energy_tot


    def _1shot_with_vrv (self, h0eff, h1eff, h2, hci_f_abq, h_qq,
                         ci, norb_f, nelec_f, orbsym=None,
                         **kwargs):
        ham_pq = np.zeros ((h_qq.shape[0]+1, h_qq.shape[1]+1), dtype=h_qq.dtype)
        ham_pq[1:,1:] = h_qq
        
        nj = np.cumsum (norb_f)
        ni = nj - norb_f
        zipper = [h0eff, h1eff, hci_f_abq, ci, norb_f, nelec_f, self.fcisolvers, ni, nj]
        e1 = []
        ci1 = []
        e_q, si_q = linalg.eigh (h_qq)
        i, j = ni[1], nj[1]
        for h0e, h1e, hc_abq, c, no, ne, solver, i, j in zip (*zipper):
            nelec = self._get_nelec (solver, ne)
            h2e = h2[i:j,i:j,i:j,i:j]
            vrv_qab = np.tensordot (si_q, hc_abq, axes=((0,),(-1,)))
            vrvsolver = vrv_fcisolver (solver, vrv_qab, e_q)
            e, c1 = vrvsolver.kernel (h1e, h2e, no, nelec, ci0=c, ecore=h0e,
                orbsym=orbsym, **kwargs)
            e1.append (e) 
            ci1.append (c1)
        return e1, ci1

    def get_ham_pq (self, h0, h1, h2, ci_p):
        active_frags = self.active_frags
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
            ham_pq, _, ovlp_pq = op[self.opt].ham (self, h1, h2, ci_fr, nelec_frs, soc=0,
                                                   orbsym=self.orbsym_ref, wfnsym=self.wfnsym_ref)
        return ham_pq + (h0*ovlp_pq)

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
        hci_f_abq = [hc[0][0] for hc in hci_fr_pabq]
        return hci_f_abq

    def prepare_vrvsolvers_(self, h0, h1, h2):
        norb_f = np.asarray ([self.norb_ref[ifrag] for ifrag in self.active_frags])
        nelec_f = np.asarray ([self.nelec_ref[ifrag] for ifrag in self.active_frags])
        ci0 = self._check_init_guess (None, norb_f, nelec_f, h1, h2)

        # Find lowest-energy ENV, including VRV contributions
        lroots = get_lroots (ci0)
        ham_pq = self.get_ham_pq (h0, h1, h2, ci0)
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

        vrvsolvers = []
        for ix, solver in enumerate (self.fcisolvers):
            vrvsolvers.append (vrv_fcisolver (solver, None, e_q))
        return ci0, vrvsolvers, e_q, si_q

    def revert_vrvsolvers_(self):
        for ix, solver in enumerate (self.fcisolvers):
            self.fcisolvers[ix] = solver.base
        self._e_q = []
        self._si_q = []

class VRVDressedFCISolver (object):
    def contract_vrv (self, ket):
        vrv_qab, denom_q = self.vrv_qab, self.denom_q
        if vrv_qab is None: return 0
        ket_shape = ket.shape
        idx = np.abs (denom_q) > 1e-16
        nq = np.count_nonzero (idx)
        if not nq: return 0
        vrv_qab, denom_q = vrv_qab[idx].reshape (nq,-1), denom_q[idx]
        vrv_q = np.dot (vrv_qab.conj (), ket.ravel ()) / denom_q
        hket = np.dot (vrv_q, vrv_qab).reshape (ket_shape)
        return hket
        
def vrv_fcisolver (fciobj, vrv_qab, e_q):
    if isinstance (fciobj, VRVDressedFCISolver):
        fciobj.vrv_qab = vrv_qab
        fciobj.e_q = e_q
        return fciobj
    # Should be injected below the state-averaged layer
    if isinstance (fciobj, StateAverageFCISolver):
        fciobj_class = fciobj._base_class
        weights = fciobj.weights
    else:
        fciobj_class = fciobj.__class__
        weights = None
    class FCISolver (fciobj_class, VRVDressedFCISolver):
        def __init__(self, fcibase, my_vrv, my_eq):
            self.base = copy.copy (fcibase)
            self.__dict__.update (fcibase.__dict__)
            keys = set (('contract_vrv', 'base', 'vrv_qab', 'denom_q', 'e_q'))
            self.denom_q = 0
            self.e_q = my_eq
            self.vrv_qab = my_vrv
            self._keys = self._keys.union (keys)
            self.davidson_only = self.base.davidson_only = True
        def contract_2e(self, eri, fcivec, norb, nelec, link_index=None, **kwargs):
            ci0 = self.base_contract_2e (eri, fcivec, norb, nelec, link_index, **kwargs)
            ci0 += self.contract_vrv (fcivec)
            return ci0
        def base_contract_2e (self, *args, **kwargs):
            return fciobj_class.contract_2e (self, *args, **kwargs)
        def kernel (self, h1e, h2e, norb, nelec, ecore=0, ci0=None, orbsym=None, **kwargs):
            # converge on e0
            max_cycle_e0 = 100
            conv_tol_e0 = 1e-8
            e0_last = 0
            e0 = ecore
            converged = False
            if ci0 is not None:
                c0 = np.asarray (ci0)
                if c0.ndim > 2: c0 = c0[0]
                h2eff = self.absorb_h1e (h1e, h2e, norb, nelec, 0.5)
                hc0 = self.contract_2e (h2eff, c0, norb, nelec)
                e0 += np.dot (c0.conj ().ravel (), hc0.ravel ())
            e0_first = e0
            ci1 = ci0
            for it in range (max_cycle_e0):
                self.denom_q = e0 - self.e_q
                e, ci1 = fciobj_class.kernel (
                    self, h1e, h2e, norb, nelec, ecore=ecore, ci0=ci1, orbsym=orbsym, **kwargs
                )
                e0_last = e0
                e0 = e[0] if isinstance (e, (list,tuple,np.ndarray)) else e
                ci0 = ci1
                if abs(e0-e0_last)<conv_tol_e0:
                    converged = True
                    break
            self.converged = converged and self.converged
            return e, ci1

    new_fciobj = FCISolver (fciobj, vrv_qab, e_q)
    if weights is not None: new_fciobj = state_average_fcisolver (new_fciobj, weights=weights)
    return new_fciobj


