import numpy as np
import itertools
from mrh.util.basis import represent_operator_in_basis, project_operator_into_subspace

def debug_ofc_oneRDM ( dmet_obj ):

    # Make sure its idempotent part is what it is supposed to be
    oneRDM_loc = dmet_obj.helper.construct1RDM_loc( dmet_obj.doSCF, dmet_obj.umat )
    evals, evecs = np.linalg.eigh (oneRDM_loc)
    idem_idx = np.logical_or (np.isclose (evals, 2.0 * np.ones (dmet_obj.ints.norbs_tot)), np.isclose (evals, np.zeros (dmet_obj.ints.norbs_tot)))
    corr_idx = np.logical_not (idem_idx)
    print ("oneRDM_loc eigenvalues: {0}\neigenvectors:\n{1}".format (evals, evecs))
    print ("Testing guide 1RDM: found {0} idempotent states of {1} expected".format(
        np.count_nonzero (idem_idx), dmet_obj.ints.loc2idem.shape[1]))
    loc2tidem = evecs[:,idem_idx]
    l2t = np.asmatrix (loc2tidem)
    l2i = np.asmatrix (dmet_obj.ints.loc2idem)
    t2l = l2t.H
    i2l = l2i.H
    Pidem_loc  = np.asarray (l2i * i2l)
    Ptidem_loc = np.asarray (l2t * t2l)
    print ("error matrix of idempotent states (norm = {1}):\n{0}".format (
        Ptidem_loc - Pidem_loc, np.linalg.norm (Ptidem_loc - Pidem_loc)))
    loc2tcorr = evecs[:,corr_idx]
    l2t = np.asmatrix (loc2tcorr)
    l2c = np.asmatrix (np.concatenate ([frag.loc2as for frag in dmet_obj.fragments], axis=1))
    t2l = l2t.H
    c2l = l2c.H
    Pcorr_loc  = np.asarray (l2c * c2l)
    Ptcorr_loc = np.asarray (l2t * t2l)
    print ("error matrix of correlated states (norm = {1}):\n{0}".format (
        Ptcorr_loc - Pcorr_loc, np.linalg.norm (Ptcorr_loc - Pcorr_loc)))

def debug_Eimp (dmet_obj, frag):
    E0 = dmet_obj.ints.const ()
    E1 = 0.0
    E2 = 0.0

    OEI = dmet_obj.ints.activeOEI
    TEI = frag.impham_TEI
    JK  = dmet_obj.ints.loc_rhf_jk_bis (frag.oneRDM_loc)
    OEI_eff = OEI + (0.5 * JK)
    return (       dmet_obj.ints.const ()
                 + np.einsum ('ij,ij->', OEI_eff, frag.oneRDM_loc)
          + (0.5 * np.einsum ('ijkl,ijkl->', TEI, frag.twoRDMR_imp)))

def debug_Etot (dmet_obj):
    if dmet_obj.CC_E_TYPE == 'CASCI':
        print ("debug_Etot :: CASCI calculation; passing to debug_Eimp")
        return debug_Eimp (dmet_obj, dmet_obj.fragments[0])
    frags = dmet_obj.fragments 
    E0 = dmet_obj.ints.const ()
    E1 = 0.0
    E2 = 0.0
    print ("debug_Etot :: constant = {0}".format (E0))

    for f in itertools.product (frags, frags):
        fname = "{0} + {1}".format (f[0].frag_name, f[1].frag_name) 
        OEI    = dmet_obj.ints.activeOEI
        OEI    = represent_operator_in_basis (OEI,    bra1_basis=f[0].loc2frag, ket1_basis=f[1].loc2frag)
        oneRDM = 0.5 * sum([i.oneRDM_loc for i in f])
        oneRDM = represent_operator_in_basis (oneRDM, bra1_basis=f[0].loc2frag, ket1_basis=f[1].loc2frag)
        E1_f   = np.einsum ('ij,ij->', OEI, oneRDM)
        E1    += E1_f
        print ("debug_Etot :: fragments {0} E1 = {1}".format (fname, E1_f))
    print ("debug_Etot :: one-body = {0}".format (E1))

    for f in itertools.product (frags, frags, frags, frags):
        fname = "{0} + {1} + {2} + {3}".format (f[0].frag_name, f[1].frag_name, f[2].frag_name, f[3].frag_name) 
        TEI = dmet_obj.ints.general_tei ([i.loc2frag for i in f])
        twoRDM = 0.25 * sum([represent_operator_in_basis (i.get_twoRDM_imp (),  
            bra1_basis=np.dot (i.imp2loc, f[0].loc2frag), 
            ket1_basis=np.dot (i.imp2loc, f[1].loc2frag), 
            bra2_basis=np.dot (i.imp2loc, f[2].loc2frag), 
            ket2_basis=np.dot (i.imp2loc, f[3].loc2frag)) # MULLIKEN! ORDER!   NOT DIRAC ORDER!   NEVER DIRAC ORDER!
            for i in f])
        E2_f = 0.5 * np.einsum ('ijkl,ijkl->', TEI, twoRDM)
        E2  += E2_f
        print ("debug_Etot :: fragments {0} E2 = {1}".format (fname, E2_f))
    print ("debug_Etot :: two-body = {0}".format (E2))

    return E0 + E1 + E2


