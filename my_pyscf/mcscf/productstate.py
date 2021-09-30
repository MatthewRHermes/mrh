import numpy as np
from scipy import linalg
from pyscf import lib
from mrh.my_pyscf.mcscf.addons import StateAverageNMixFCISolver
from itertools import combinations

# TODO: linkstr support
class ProductStateFCISolver (StateAverageNMixFCISolver, lib.StreamObject):

    def __init__(self, fcisolvers, stdout=None, verbose=0, **kwargs):
        self.fcisolvers = fcisolvers
        self.verbose = verbose
        self.stdout = stdout
        self.log = lib.logger.new_logger (self, verbose)

    def kernel (self, h1, h2, norb_f, nelec_f, ecore=0, ci0=None, orbsym=None,
            conv_tol_grad=1e-4, conv_tol_self=1e-6, max_cycle_macro=50,
            **kwargs):
        log = self.log
        converged = False
        e_sigma = conv_tol_self + 1
        ci1 = ci0 # TODO: get_init_guess
        log.info ('Entering product-state fixed-point CI iteration')
        for it in range (max_cycle_macro):
            h1eff, h0eff = self.project_hfrag (h1, h2, ci1, norb_f, nelec_f,
                ecore=ecore, **kwargs)
            grad = self._get_grad (h1eff, h2, ci1, norb_f, nelec_f, **kwargs)
            grad_max = np.amax (np.abs (grad))
            log.info ('Cycle %d: max grad = %e ; sigma = %e', it, grad_max,
                e_sigma)
            if ((grad_max < conv_tol_grad) and (e_sigma < conv_tol_self)):
                converged = True
                break
            e, ci1 = self._1shot (h0eff, h1eff, h2, ci1, norb_f, nelec_f,
                orbsym=orbsym, **kwargs)
            e_sigma = np.amax (e) - np.amin (e)
        conv_str = ['NOT converged','converged'][int (converged)]
        log.info (('Product_state fixed-point CI iteration {} after {} '
                   'cycles').format (conv_str, it))
        energy_elec = self.energy_elec (h1, h2, ci1, norb_f, nelec_f,
            ecore=ecore, **kwargs)
        return converged, energy_elec, ci1


    def _1shot (self, h0eff, h1eff, h2, ci, norb_f, nelec_f, orbsym=None,
            **kwargs):
        nj = np.cumsum (norb_f)
        ni = nj - norb_f
        zipper = [h0eff, h1eff, ci, norb_f, nelec_f, self.fcisolvers, ni, nj]
        e1 = []
        ci1 = []
        for h0e, h1e, c, no, ne, solver, i, j in zip (*zipper):
            h2e = h2[i:j,i:j,i:j,i:j]
            osym = getattr (solver, 'orbsym', None)
            if orbsym is not None: osym=orbsym[i:j]
            nelec = self._get_nelec (solver, ne)
            e, c1 = solver.kernel (h1e, h2e, no, nelec, ci0=c, ecore=h0e,
                orbsym=osym, **kwargs)
            e1.append (e)
            ci1.append (c1)
        return e1, ci1

    def _get_grad (self, h1eff, h2, ci, norb_f, nelec_f, **kwargs):
        nj = np.cumsum (norb_f)
        ni = nj - norb_f
        zipper = [h1eff, ci, norb_f, nelec_f, self.fcisolvers, ni, nj]
        grad = []
        for h1e, c, no, ne, solver, i, j in zip (*zipper):
            nelec = self._get_nelec (solver, ne)
            h2e = h2[i:j,i:j,i:j,i:j]
            h2e = solver.absorb_h1e (h1e, h2e, no, nelec, 0.5)
            hc = solver.contract_2e (h2e, c, no, nelec)
            chc = c.ravel ().dot (hc.ravel ())
            hc -= c * chc
            grad.append (hc.ravel ())
        return np.concatenate (grad)

    def energy_elec (self, h1, h2, ci, norb_f, nelec_f, ecore=0, **kwargs):
        dm1 = np.stack (self.make_rdm1 (ci, norb_f, nelec_f), axis=0)
        dm2 = self.make_rdm2 (ci, norb_f, nelec_f)
        energy_tot = (ecore + np.tensordot (h1, dm1, axes=2)
                        + 0.5*np.tensordot (h2, dm2, axes=4))
        return energy_tot

    def project_hfrag (self, h1, h2, ci, norb_f, nelec_f, ecore=0, **kwargs):
        dm1s = np.stack (self.make_rdm1s (ci, norb_f, nelec_f), axis=0)
        dm1 = dm1s.sum (0)
        dm2 = self.make_rdm2 (ci, norb_f, nelec_f)
        energy_tot = (ecore + np.tensordot (h1, dm1, axes=2)
                        + 0.5*np.tensordot (h2, dm2, axes=4))
        v1  = np.tensordot (dm1s, h2, axes=2)
        v1 += v1[::1] # ja + jb
        v1 -= np.tensordot (dm1s, h2, axes=((1,2),(2,1)))
        f1 = h1[None,:,:] + v1
        h1eff = []
        h0eff = []
        nj = np.cumsum (norb_f)
        ni = nj - norb_f
        for i, j in zip (ni, nj):
            dm1s_i = dm1s[:,i:j,i:j]
            dm2_i = dm2[i:j,i:j,i:j,i:j]
            # v1_self
            h2_i = h2[i:j,i:j,:,:]
            v1_i = np.tensordot (dm1s_i, h2_i, axes=2)
            v1_i += v1_i[::-1] # ja + jb
            h2_i = h2[:,i:j,i:j,:]
            v1_i -= np.tensordot (dm1s_i, h2_i, axes=((1,2),(2,1)))
            # cancel off-diagonal energy double-counting
            e_i = energy_tot - np.tensordot (dm1s, v1_i, axes=3) # overcorrects
            # cancel h1eff double-counting
            v1_i = v1_i[:,i:j,i:j] 
            h1eff.append (f1[:,i:j,i:j]-v1_i)
            # cancel diagonal energy double-counting
            h1_i = h1[None,i:j,i:j] - v1_i # v1_i fixes overcorrect
            h2_i = h2[i:j,i:j,i:j,i:j]
            e_i -= (np.tensordot (h1_i, dm1s_i, axes=3)
              + 0.5*np.tensordot (h2_i, dm2_i, axes=4))
            h0eff.append (e_i)
        return h1eff, h0eff

    def make_rdm1s (self, ci, norb_f, nelec_f, **kwargs):
        norb = sum (norb_f)
        dm1a = np.zeros ((norb, norb))
        dm1b = np.zeros ((norb, norb))
        nj = np.cumsum (norb_f)
        ni = nj - norb_f
        for i, j, c, no, ne, s in zip (ni, nj, ci, norb_f, nelec_f, self.fcisolvers):
            nelec = self._get_nelec (s, ne)
            a, b = s.make_rdm1s (c, no, nelec)
            dm1a[i:j,i:j] = a[:,:]
            dm1b[i:j,i:j] = b[:,:]
        return dm1a, dm1b

    def make_rdm1 (self, ci, norb_f, nelec_f, **kwargs):
        dm1a, dm1b = self.make_rdm1s (ci, norb_f, nelec_f, **kwargs)
        return dm1a + dm1b

    def make_rdm2 (self, ci, norb_f, nelec_f, **kwargs):
        norb = sum (norb_f)
        dm2 = np.zeros ([norb,]*4)
        nj = np.cumsum (norb_f)
        ni = nj - norb_f
        dm1a, dm1b = self.make_rdm1s (ci, norb_f, nelec_f, **kwargs)
        for i, j, c, no, ne, s in zip (ni, nj, ci, norb_f, nelec_f, self.fcisolvers):
            nelec = self._get_nelec (s, ne)
            dm2[i:j,i:j,i:j,i:j] = s.make_rdm2 (c, no, nelec)
        dm1 = dm1a + dm1b
        for (i,j), (k,l) in combinations (zip (ni, nj), 2):
            d1_ij, d1a_ij, d1b_ij = dm1[i:j,i:j], dm1a[i:j,i:j], dm1b[i:j,i:j]
            d1_kl, d1a_kl, d1b_kl = dm1[k:l,k:l], dm1a[k:l,k:l], dm1b[k:l,k:l]
            d2 = np.multiply.outer (d1_ij, d1_kl)
            dm2[i:j,i:j,k:l,k:l] = d2
            dm2[k:l,k:l,i:j,i:j] = d2.transpose (2,3,0,1)
            d2  = np.multiply.outer (d1a_ij, d1a_kl)
            d2 += np.multiply.outer (d1b_ij, d1b_kl)
            dm2[i:j,k:l,k:l,i:j] = -d2.transpose (0,2,3,1)
            dm2[k:l,i:j,i:j,k:l] = -d2.transpose (2,0,1,3)
        return dm2

