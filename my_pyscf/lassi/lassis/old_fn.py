    def get_hpp_xp (self, h1, h2, ci0, si_p, norb_f, nelec_f, ecore=0, nroots=1, **kwargs):
        nfrag = len (ci0)
        h1eff = [[] for i in range (nfrag)]
        h0eff = []
        lroots = get_lroots (ci0)
        assert (len (lroots) == 2)
        assert (lroots[0] == lroots[1]), '{} {}'.format (lroots, nroots)
        nroots = lroots[0]
        for iroot in range (nroots):
            c = [x[iroot] for x in ci0]
            h1e, h0e = self.project_hfrag (h1, h2, c, norb_f, nelec_f, ecore=ecore, **kwargs)[:2]
            for ifrag in range (nfrag):
                h1eff[ifrag].append (h1e[ifrag])
            h0eff.append (h0e)
        h0eff = np.asarray (h0eff).T
        nj = np.cumsum (norb_f)
        ni = nj - norb_f
        zipper = [h1eff, h0eff, ci0, norb_f, nelec_f, self.fcisolvers, ni, nj, lroots]
        hci_f_pab = []
        for ifrag, (h1ef, h0ef, c, no, ne, solver, i, j, nroots) in enumerate (zip (*zipper)):
            jfrag = 1 if ifrag==0 else 0
            k, l = ni[jfrag], nj[jfrag]
            nelec = self._get_nelec (solver, ne)
            nelec_j = self._get_nelec (self.fcisolvers[jfrag], nelec_f[jfrag])
            h2e = h2[i:j,i:j,i:j,i:j]
            h2e_j = h2[i:j,i:j,k:l,k:l]
            h2e_k = h2[i:j,k:l,k:l,i:j].transpose (0,3,2,1)
            hc = []
            # Diagonal part: Cn <nKnL|H|*KnL> Cn
            for iroot, (icol, h1e, h0e, si) in enumerate (zip (c, h1ef, h0ef, si_p)):
                h2e = solver.absorb_h1e (h1e, h2e, no, nelec, 0.5)
                hcol = solver.contract_2e (h2e, icol, no, nelec) + (h0e * icol)
                hc.append (si * hcol)
                # Off-diagonal part: Cm <mKmL|H|*KnL> Cn
                for jroot, (jcol, sj) in enumerate (zip (c, si_p)):
                    if iroot==jroot: continue
                    tdm1s = trans_rdm12s (ci0[jfrag][iroot], ci0[jfrag][jroot], norb_f[jfrag], nelec_j)[0]
                    tdm1s = np.stack (tdm1s, axis=0).transpose (0, 2, 1)
                    vj = np.tensordot (h2e_j, tdm1s.sum (0), axes=2)
                    vk = np.tensordot (tdm1s, h2e_k, axes=((1,2),(2,3)))
                    veff = vj[None,:,:] - vk
                    hc[iroot] += sj * contract_1e_nosym (veff, jcol, no, nelec)
            c, hc = np.asarray (c), np.asarray (hc) 
            chc = np.dot (np.asarray (c).reshape (nroots,-1).conj (),
                          np.asarray (hc).reshape (nroots,-1).T).T
            hc = hc - np.tensordot (chc, c, axes=1)
            hci_f_pab.append (hc)
        return hci_f_pab
