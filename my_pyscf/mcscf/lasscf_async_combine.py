import numpy as np
from scipy import linalg
from pyscf.lo import orth
from pyscf.scf.rohf import get_roothaan_fock
from mrh.my_pyscf.mcscf import lasci_sync

# TODO: symmetry
def orth_orb (las, kf2_list):
    ncore, ncas = las.ncore, las.ncas
    nocc = ncore + ncas
    nfrags = len (kf2_list)

    # orthonormalize active orbitals
    mo_coeff = las.mo_coeff.copy ()
    ci = []
    for ifrag, kf2 in enumerate (kf2_list):
        i = las.ncore + sum (las.ncas_sub[:ifrag])
        j = i + las.ncas_sub[ifrag]
        mo_coeff[:,i:j] = kf2.mo_coeff[:,i:j]
        ci.append (kf2.ci[ifrag])
    mo_cas_preorth = mo_coeff[:,ncore:nocc].copy ()
    s0 = las._scf.get_ovlp ()
    mo_coeff[:,ncore:nocc] = orth.vec_lowdin (mo_cas_preorth, s=s0)

    # reassign orthonormalized active orbitals
    proj = []
    for ifrag in range (nfrags):
        i = sum (las.ncas_sub[:ifrag])
        j = i + las.ncas_sub[ifrag]
        proj.append (mo_cas_preorth[:,i:j] @ mo_cas_preorth[:,i:j].conj ().T)
    smo1 = s0 @ mo_coeff[:,ncore:nocc]
    frag_weights = np.stack ([((p @ smo1) * smo1.conjugate ()).sum (0)
                              for p in proj, axis=-1)
    idx = np.argsort (frag_weights, axis=1)[:,-1]
    mo_las = []
    for ifrag in range (nfrags):
        mo = mo_cas[:,(idx == ifrag)]
        i = sum (las.ncas_sub[:ifrag])
        j = i + las.ncas_sub[ifrag]
        s1 = mo.conj ().T @ s0 @ mo_cas_preorth[:,i:j]
        u, svals, vh = linalg.svd (s1)
        mo_las.append (mo @ u @ vh)
    mo_cas = np.concatenate (mo_las, axis=1)
    
    # non-active orbitals
    mo1 = np.concatenate ([mo_cas, mo_coeff[:,:ncore], mo_coeff[:,nocc:]], axis=1)
    mo1 = orth.vec_schimdt (mo1, s=s0)[:,ncas:]
    veff = sum ([kf2.veff for kf2 in kf2_list]) / nfrags
    dm1s = sum ([kf2.dm1s for dm1s in kf2_list]) / nfrags
    fock = las.get_hcore ()[None,:,:] + veff
    fock = get_roothaan_fock (fock, dm1s, s0)
    orbsym = None # TODO: symmetry
    fock = mo1.conj ().T @ fock @ mo1
    ene, umat = las._eig (fock, 0, 0, orbsym)
    mo_core = mo1 @ umat[:,:ncore]
    mo_virt = mo1 @ umat[:,ncore:]
    mo_coeff = np.concatenate ([mo_core, mo_cas, mo_virt], axis=1)

    return las.get_keyframe (mo_coeff, ci)

def relax (las, kf):

    converged, e_tot, e_states, mo_energy, mo_coeff, e_cas, ci, h2eff_sub, veff = \
        lasci_sync.kernel (self, kf.mo_coeff, ci0=kf.ci)

    return las.get_keyframe (mo_coeff, ci)

def combine_o0 (las, kf2_list):
    kf1 = orth_orb (las, kf2_list)
    kf1 = relax (las, kf1)
    return kf1


