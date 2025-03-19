import numpy as np

def orbital_response(ppaa, papa, eri_paaa, ocm2, tcm2, gorb, ncore, nocc, nmo):
    f1_prime = np.zeros((nmo, nmo), dtype=np.float64)
    for p, f1 in enumerate (f1_prime):
        praa = ppaa[p]
        para = papa[p]
        paaa = praa[ncore:nocc]
        # g_pabc d_qabc + g_prab d_qrab + g_parb d_qarb + g_pabr d_qabr (Formal)
        #        d_cbaq          d_abqr          d_aqbr          d_qabr (Symmetry of ocm2)
        # g_pcba d_abcq + g_prab d_abqr + g_parc d_aqcr + g_pbcr d_qbcr (Relabel)
        #                                                 g_pbrc        (Symmetry of eri)
        # g_pcba d_abcq + g_prab d_abqr + g_parc d_aqcr + g_pbrc d_qbcr (Final)
        for i, j in ((0, ncore), (nocc, nmo)): # Don't double-count
            ra, ar, cm = praa[i:j], para[:,i:j], ocm2[:,:,:,i:j]
            f1[i:j] += np.tensordot (paaa, cm, axes=((0,1,2),(2,1,0))) # last index external
            f1[ncore:nocc] += np.tensordot (ra, cm, axes=((0,1,2),(3,0,1))) # third index external
            f1[ncore:nocc] += np.tensordot (ar, cm, axes=((0,1,2),(0,3,2))) # second index external
            f1[ncore:nocc] += np.tensordot (ar, cm, axes=((0,1,2),(1,3,2))) # first index external

    # (H.x_aa)_va, (H.x_aa)_ac
    ocm2 = ocm2[:,:,:,ncore:nocc] + ocm2[:,:,:,ncore:nocc].transpose (1,0,3,2)
    ocm2 += ocm2.transpose (2,3,0,1)
    ecm2 = ocm2 + tcm2
    
    f1_prime[:ncore,ncore:nocc] += np.tensordot (eri_paaa[:ncore], ecm2, axes=((1,2,3),(1,2,3)))
    f1_prime[nocc:,ncore:nocc] += np.tensordot (eri_paaa[nocc:], ecm2, axes=((1,2,3),(1,2,3)))

    return gorb + (f1_prime - f1_prime.T)

def orbital_response_debug(ppaa, papa, eri_paaa, ocm2, tcm2, gorb, ncore, nocc, nmo):
    f1_prime = np.zeros((nmo, nmo), dtype=np.float64)
    for p, f1 in enumerate (f1_prime):
        praa = ppaa[p]
        para = papa[p]
        paaa = praa[ncore:nocc]
        # g_pabc d_qabc + g_prab d_qrab + g_parb d_qarb + g_pabr d_qabr (Formal)
        #        d_cbaq          d_abqr          d_aqbr          d_qabr (Symmetry of ocm2)
        # g_pcba d_abcq + g_prab d_abqr + g_parc d_aqcr + g_pbcr d_qbcr (Relabel)
        #                                                 g_pbrc        (Symmetry of eri)
        # g_pcba d_abcq + g_prab d_abqr + g_parc d_aqcr + g_pbrc d_qbcr (Final)
        i = 0
        j = ncore
        ra, ar, cm = praa[i:j], para[:,i:j], ocm2[:,:,:,i:j]
        f1[i:j] += np.tensordot (paaa, cm, axes=((0,1,2),(2,1,0))) # last index external
        f1[ncore:nocc] += np.tensordot (ra, cm, axes=((0,1,2),(3,0,1))) # third index external
        f1[ncore:nocc] += np.tensordot (ar, cm, axes=((0,1,2),(0,3,2))) # second index external
        f1[ncore:nocc] += np.tensordot (ar, cm, axes=((0,1,2),(1,3,2))) # first index external


        i = nocc
        j = nmo
        ra, ar, cm = praa[i:j], para[:,i:j], ocm2[:,:,:,i:j]
        f1[i:j] += np.tensordot (paaa, cm, axes=((0,1,2),(2,1,0))) # last index external
        f1[ncore:nocc] += np.tensordot (ra, cm, axes=((0,1,2),(3,0,1))) # third index external
        f1[ncore:nocc] += np.tensordot (ar, cm, axes=((0,1,2),(0,3,2))) # second index external
        f1[ncore:nocc] += np.tensordot (ar, cm, axes=((0,1,2),(1,3,2))) # first index external

    # (H.x_aa)_va, (H.x_aa)_ac
    ocm2 = ocm2[:,:,:,ncore:nocc] + ocm2[:,:,:,ncore:nocc].transpose (1,0,3,2)
    ocm2 += ocm2.transpose (2,3,0,1)
    ecm2 = ocm2 + tcm2
    
    f1_prime[:ncore,ncore:nocc] += np.tensordot (eri_paaa[:ncore], ecm2, axes=((1,2,3),(1,2,3)))
    f1_prime[nocc:,ncore:nocc] += np.tensordot (eri_paaa[nocc:], ecm2, axes=((1,2,3),(1,2,3)))
    
    return gorb + (f1_prime - f1_prime.T)
