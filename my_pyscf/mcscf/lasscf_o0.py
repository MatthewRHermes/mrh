import numpy as np
from scipy import linalg
from mrh.util.la import matrix_svd_control_options
from mrh.my_pyscf.mcscf import lasci, lasci_sync, _DFLASCI
from pyscf import gto, scf, symm
from pyscf.mcscf import mc_ao2mo, casci_symm, mc1step
from pyscf.mcscf import df as mc_df
from pyscf.lo import orth
from pyscf.lib import tag_array
from functools import partial

# An implementation that carries out vLASSCF, but without utilizing Schmidt decompositions
# or "fragment" subspaces, so that the orbital-optimization part scales no better than
# CASSCF. Eventually to be modified into a true all-PySCF implementation of vLASSCF

def localize_init_guess (las, frags_orbs, mo_coeff, spin, lo_coeff, fock, ao_ovlp, freeze_cas_spaces=False):
    ''' Project active orbitals into sets of orthonormal "fragments" defined by lo_coeff
    and frags_orbs, and orthonormalize inactive and virtual orbitals in the orthogonal complement
    space. Beware that unless freeze_cas_spaces=True, frozen orbitals will not be preserved.

    Args:
        las: LASSCF or LASCI object
        frags_orbs: list of length nfrags
            Contains list of AO indices formally defining the fragments
            into which the active orbitals are to be localized

    Kwargs: (some of these are args here but kwargs in the actual caller)
        mo_coeff: ndarray of shape (nao, nmo)
            Molecular orbital coefficients containing active orbitals
            on columns ncore:ncore+ncas
        spin: integer
            Unused; retained for backwards compatibility I guess
        lo_coeff: ndarray of shape (nao, nao)
            Linear combinations of AOs that are localized and orthonormal
        fock: ndarray of shape (nmo, nmo)
            Effective 1-electron Hamiltonian matrix for recanonicalizing
            the inactive and external sectors after the latter are
            possibly distorted by the projection of the active orbitals
        ao_ovlp: ndarray of shape (nao, nao)
            Overlap matrix of the underlying AO basis
        freeze_cas_spaces: logical
            If true, then active orbitals are mixed only among themselves
            when localizing, which leaves the inactive and external sectors
            unchanged (to within numerical precision). Otherwise, active
            orbitals are projected into the localized-orbital space and
            the inactive and external orbitals are reconstructed as closely
            as possible using SVD.

    Returns:
        mo_coeff: ndarray of shape (nao,nmo)
            Orbital coefficients after localization of the active space;
            columns in the order (inactive,las1,las2,...,lasn,external)
    '''
    # For reasons that pass my understanding, mo_coeff sometimes can't be assigned symmetry
    # by PySCF's own code. Therefore, I'm going to keep the symmetry tags on mo_coeff
    # and make sure the SVD engine sees them and doesn't try to figure it out itself.
    # Hopefully this never becomes a problem with the lo_coeff.
    ncore, ncas, ncas_sub = las.ncore, las.ncas, las.ncas_sub
    nocc = ncore + ncas
    nfrags = len (frags_orbs)
    nao, nmo = mo_coeff.shape
    unused_aos = np.ones (nao, dtype=np.bool_)
    for frag_orbs in frags_orbs: unused_aos[frag_orbs] = False
    has_orbsym = hasattr (mo_coeff, 'orbsym')
    mo_orbsym = getattr (mo_coeff, 'orbsym', np.zeros (nmo))
    mo_coeff = mo_coeff.copy () # Safety

    # SVD to pick active orbitals
    mo_cas = tag_array (mo_coeff[:,ncore:nocc], orbsym=mo_orbsym[ncore:nocc])
    if freeze_cas_spaces:
        null_coeff = np.hstack ([mo_coeff[:,:ncore], mo_coeff[:,nocc:]])
    else:
        null_coeff = lo_coeff[:,unused_aos]
    for ix, (nlas, frag_orbs) in enumerate (zip (las.ncas_sub, frags_orbs)):
        try:
            mo_proj, sval, mo_cas = las._svd (lo_coeff[:,frag_orbs], mo_cas, s=ao_ovlp)
        except ValueError as e:
            print (ix, lo_coeff[:,frag_orbs].shape, ao_ovlp.shape, mo_cas.shape)
            print (mo_cas.orbsym)
            raise (e)
        i, j = ncore + sum (las.ncas_sub[:ix]), ncore + sum (las.ncas_sub[:ix]) + nlas
        mo_las = mo_cas if freeze_cas_spaces else mo_proj
        mo_coeff[:,i:j] = mo_las[:,:nlas]
        if has_orbsym: mo_orbsym[i:j] = mo_las.orbsym[:nlas]
        if freeze_cas_spaces:
            if has_orbsym: orbsym = mo_cas.orbsym[nlas:]
            mo_cas = mo_cas[:,nlas:]
            if has_orbsym: mo_cas = tag_array (mo_cas, orbsym=orbsym)
        else:
            null_coeff = np.hstack ([null_coeff, mo_proj[:,nlas:]])

    # SVD of null space to pick inactive orbitals
    assert (null_coeff.shape[-1] + ncas == nmo)
    mo_core = tag_array (mo_coeff[:,:ncore], orbsym=mo_orbsym[:ncore])
    mo_proj, sval, mo_core = las._svd (null_coeff, mo_core, s=ao_ovlp)
    mo_coeff[:,:ncore], mo_coeff[:,nocc:] = mo_proj[:,:ncore], mo_proj[:,ncore:]
    if has_orbsym:
        mo_orbsym[:ncore] = mo_proj.orbsym[:ncore]
        mo_orbsym[nocc:] = mo_proj.orbsym[ncore:]
    mo_coeff = tag_array (mo_coeff, orbsym=mo_orbsym)

    # Canonicalize for good init CI guess and visualization
    ranges = [(0,ncore),(nocc,nmo)]
    for ix, di in enumerate (ncas_sub):
        i = sum (ncas_sub[:ix])
        ranges.append ((i,i+di))
    fock = mo_coeff.conj ().T @ fock @ mo_coeff
    for i, j in ranges:
        if (j == i): continue
        e, c = las._eig (fock[i:j,i:j], i, j)
        idx = np.argsort (e)
        mo_coeff[:,i:j] = mo_coeff[:,i:j] @ c[:,idx]
        mo_orbsym[i:j] = mo_orbsym[i:j][idx]
    if has_orbsym: mo_coeff = tag_array (mo_coeff, orbsym=mo_orbsym)
    else: mo_coeff = np.array (mo_coeff) # remove spurious tag
    return mo_coeff


class LASSCF_UnitaryGroupGenerators (lasci_sync.LASCI_UnitaryGroupGenerators):

    def _init_orb (self, las, mo_coeff, ci):
        lasci_sync.LASCI_UnitaryGroupGenerators._init_nonfrozen_orb (self, las)
        self.uniq_orb_idx = self.nfrz_orb_idx.copy ()
        # The distinction between "uniq_orb_idx" and "nfrz_orb_idx" is an
        # artifact of backwards-compatibility with the old LASSCF implementation

class LASSCFSymm_UnitaryGroupGenerators (LASSCF_UnitaryGroupGenerators):
    __init__ = lasci_sync.LASCISymm_UnitaryGroupGenerators.__init__
    _init_ci = lasci_sync.LASCISymm_UnitaryGroupGenerators._init_ci
    def _init_orb (self, las, mo_coeff, ci, orbsym):
        LASSCF_UnitaryGroupGenerators._init_orb (self, las, mo_coeff, ci)
        self.symm_forbid = (orbsym[:,None] ^ orbsym[None,:]).astype (np.bool_)
        self.uniq_orb_idx[self.symm_forbid] = False
        self.nfrz_orb_idx[self.symm_forbid] = False

class LASSCF_HessianOperator (lasci_sync.LASCI_HessianOperator):
    # Required modifications for Hx: [I forgot about 3) at first]
    #   1) cache CASSCF-type eris and paaa - init_eri
    #   2) increase range of ocm2 - make_odm1s2c_sub
    #   3) extend veff to active-unactive sector - split_veff
    #   4) dot the above three together - orbital_response
    #   5) TODO: get_veff using DF needs to be extended as well
    # Required modifications for API:
    #   6) broader ERI rotation - update_mo_ci_eri
    # Possible modifications:
    #   7) current prec may not be "good enough" - get_prec
    #   8) define "gx" in this context - get_gx 

    def _init_eri_(self):
        lasci_sync._init_df_(self)
        if isinstance (self.las, _DFLASCI):
            self.cas_type_eris = mc_df._ERIS (self.las, self.mo_coeff, self.with_df)
        else:
            self.cas_type_eris = mc_ao2mo._ERIS (self.las, self.mo_coeff,
                method='incore', level=2) # level=2 -> ppaa, papa only
                # level=1 computes more stuff; it's only useful if I
                # want the honest hdiag in get_prec ()
        ncore, ncas = self.ncore, self.ncas
        nocc = ncore + ncas

    def get_veff (self, dm1s_mo=None):
        mo = self.mo_coeff
        moH = mo.conjugate ().T
        nmo = mo.shape[-1]
        dm1_mo = dm1s_mo.sum (0)
        dm1_ao = np.dot (mo, np.dot (dm1_mo, moH))
        veff_ao = np.squeeze (self.las.get_veff (dm1s=dm1_ao))
        return np.dot (moH, np.dot (veff_ao, mo))

    def split_veff (self, veff_mo, dm1s_mo):
        veff_c = veff_mo.copy ()
        ncore = self.ncore
        nocc = self.nocc
        sdm = dm1s_mo[0] - dm1s_mo[1]
        sdm_ra = sdm[:,ncore:nocc]
        sdm_ar = sdm[ncore:nocc,:].copy ()
        sdm_ar[:,ncore:nocc] = 0.0
        veff_s = np.zeros_like (veff_c)
        vk_pa = veff_s[:,ncore:nocc]
        for p, v1 in enumerate (vk_pa):
            praa = self.cas_type_eris.ppaa[p]
            para = self.cas_type_eris.papa[p]
            paaa = praa[ncore:nocc]
            v1[:]  = np.tensordot (sdm_ra, praa, axes=2)
            v1[:] += np.tensordot (sdm_ar, para, axes=2)
        veff_s[:,:] *= -0.5
        vk_aa = vk_pa[ncore:nocc]
        veff_s[ncore:nocc,:] = vk_pa.T
        assert (np.allclose (veff_s, veff_s.T)), vk_aa-vk_aa.T
        veffa = veff_c + veff_s
        veffb = veff_c - veff_s
        return np.stack ([veffa, veffb], axis=0)

    def orbital_response (self, kappa, odm1s, ocm2, tdm1frs, tcm2, veff_prime):
        ''' Parent class does everything except va/ac degrees of freedom
        (c: closed; a: active; v: virtual; p: any) '''
        ncore, nocc, nmo = self.ncore, self.nocc, self.nmo
        gorb = lasci_sync.LASCI_HessianOperator.orbital_response (self, kappa, odm1s,
            ocm2, tdm1frs, tcm2, veff_prime)
        f1_prime = np.zeros ((self.nmo, self.nmo), dtype=self.dtype)
        # (H.x_va)_pp, (H.x_ac)_pp sector
        for p, f1 in enumerate (f1_prime):
            praa = self.cas_type_eris.ppaa[p]
            para = self.cas_type_eris.papa[p]
            paaa = praa[ncore:nocc]
            assert (np.allclose (paaa, self.eri_paaa[p]))
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
        f1_prime[:ncore,ncore:nocc] += np.tensordot (self.eri_paaa[:ncore], ecm2, axes=((1,2,3),(1,2,3)))
        f1_prime[nocc:,ncore:nocc] += np.tensordot (self.eri_paaa[nocc:], ecm2, axes=((1,2,3),(1,2,3)))
        return gorb + (f1_prime - f1_prime.T)

    def _update_h2eff_sub (self, mo1, umat, h2eff_sub):
        return self.las.ao2mo (mo1)

class LASSCFNoSymm (lasci.LASCINoSymm):
    _ugg = LASSCF_UnitaryGroupGenerators
    _hop = LASSCF_HessianOperator
    as_scanner = mc1step.as_scanner
    def split_veff (self, veff, h2eff_sub, mo_coeff=None, ci=None, casdm1s_sub=None): 
        # This needs to actually do the veff, otherwise the preconditioner is broken
        # Eventually I can remove this, once I've implemented Schmidt decomposition etc. etc.
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if ci is None: ci = self.ci
        if casdm1s_sub is None: casdm1s_sub = self.make_casdm1s_sub (ci = ci)
        if isinstance (self, _DFLASCI):
            get_jk = self.with_df.get_jk
        else:
            get_jk = partial (self._scf.get_jk, self.mol)
        ints = self.with_df if isinstance (self, _DFLASCI) else self._scf
        mo_cas = mo_coeff[:,self.ncore:][:,:self.ncas]
        dm1s_cas = linalg.block_diag (*[dm[0] - dm[1] for dm in casdm1s_sub])
        dm1s = mo_cas @ dm1s_cas @ mo_cas.conj ().T
        veff_c = veff.copy ()
        veff_s = -get_jk (dm1s, hermi=1)[1]/2
        veff_a = veff_c + veff_s
        veff_b = veff_c - veff_s
        veff = np.stack ([veff_a, veff_b], axis=0)
        dm1s = self.make_rdm1s (mo_coeff=mo_coeff, casdm1s_sub=casdm1s_sub)
        vj, vk = get_jk (dm1s, hermi=1)
        veff_a = vj[0] + vj[1] - vk[0]
        veff_b = vj[0] + vj[1] - vk[1]
        veff_test = np.stack ([veff_a, veff_b], axis=0)
        assert (np.allclose (veff, veff_test))
        return veff

    def localize_init_guess (self, frags_atoms, mo_coeff=None, spin=None, lo_coeff=None, fock=None, freeze_cas_spaces=False):
        ''' Here spin = 2s = number of singly-occupied orbitals '''
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if lo_coeff is None:
            lo_coeff = orth.orth_ao (self.mol, 'meta_lowdin')
        if spin is None:
            spin = self.nelecas[0] - self.nelecas[1]
        assert (spin % 2 == sum (self.nelecas) % 2)
        assert (len (frags_atoms) == len (self.ncas_sub))
        ao_offset = self.mol.offset_ao_by_atom ()
        frags_orbs = [[orb for atom in frag_atoms for orb in list (range (ao_offset[atom,2], ao_offset[atom,3]))] for frag_atoms in frags_atoms]
        if fock is None: fock = self._scf.get_fock ()
        ao_ovlp = self._scf.get_ovlp ()
        return localize_init_guess (self, frags_orbs, mo_coeff, spin, lo_coeff, fock, ao_ovlp, freeze_cas_spaces=freeze_cas_spaces)

    def _svd (self, mo_lspace, mo_rspace, s=None, **kwargs):
        if s is None: s = self._scf.get_ovlp ()
        return matrix_svd_control_options (s, lspace=mo_lspace, rspace=mo_rspace, full_matrices=True)[:3]
 
class LASSCFSymm (lasci.LASCISymm):
    _ugg = LASSCFSymm_UnitaryGroupGenerators    
    _hop = LASSCF_HessianOperator
    split_veff = LASSCFNoSymm.split_veff
    as_scanner = mc1step.as_scanner

    def localize_init_guess (self, frags_atoms, mo_coeff=None, spin=None, lo_coeff=None, fock=None, freeze_cas_spaces=False):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        mo_coeff = casci_symm.label_symmetry_(self, mo_coeff)
        return LASSCFNoSymm.localize_init_guess (self, frags_atoms, mo_coeff=mo_coeff, spin=spin,
            lo_coeff=lo_coeff, fock=fock, freeze_cas_spaces=freeze_cas_spaces)

    def _svd (self, mo_lspace, mo_rspace, s=None, **kwargs):
        if s is None: s = self._scf.get_ovlp ()
        lsymm = getattr (mo_lspace, 'orbsym', None)
        if lsymm is None:
            mo_lspace = symm.symmetrize_space (self.mol, mo_lspace)
            lsymm = symm.label_orb_symm(self.mol, self.mol.irrep_id,
                self.mol.symm_orb, mo_lspace, s=s)
        rsymm = getattr (mo_rspace, 'orbsym', None)
        if rsymm is None:
            mo_rspace = symm.symmetrize_space (self.mol, mo_rspace)
            rsymm = symm.label_orb_symm(self.mol, self.mol.irrep_id,
                self.mol.symm_orb, mo_rspace, s=s)
        decomp = matrix_svd_control_options (s,
            lspace=mo_lspace, rspace=mo_rspace, 
            lspace_symmetry=lsymm, rspace_symmetry=rsymm,
            full_matrices=True, strong_symm=True)
        mo_lvecs, svals, mo_rvecs, lsymm, rsymm = decomp
        mo_lvecs = tag_array (mo_lvecs, orbsym=lsymm)
        mo_rvecs = tag_array (mo_rvecs, orbsym=rsymm)
        return mo_lvecs, svals, mo_rvecs

def LASSCF (mf_or_mol, ncas_sub, nelecas_sub, **kwargs):
    if isinstance(mf_or_mol, gto.Mole):
        mf = scf.RHF(mf_or_mol)
    else:
        mf = mf_or_mol
    if mf.mol.symmetry: 
        las = LASSCFSymm (mf, ncas_sub, nelecas_sub, **kwargs)
    else:
        las = LASSCFNoSymm (mf, ncas_sub, nelecas_sub, **kwargs)
    if getattr (mf, 'with_df', None):
        las = lasci.density_fit (las, with_df = mf.with_df) 
    return las


if __name__ == '__main__':
    from pyscf import scf, lib, tools, mcscf
    import os
    class cd:
        """Context manager for changing the current working directory"""
        def __init__(self, newPath):
            self.newPath = os.path.expanduser(newPath)

        def __enter__(self):
            self.savedPath = os.getcwd()
            os.chdir(self.newPath)

        def __exit__(self, etype, value, traceback):
            os.chdir(self.savedPath)
    from mrh.tests.lasscf.me2n2_struct import structure as struct
    from mrh.my_pyscf.fci import csf_solver
    with cd ("/home/herme068/gits/mrh/tests/lasscf"):
        mol = struct (2.0, '6-31g')
    mol.output = 'lasscf_o0.log'
    mol.verbose = lib.logger.DEBUG
    mol.build ()
    mf = scf.RHF (mol).run ()
    mo_coeff = mf.mo_coeff.copy ()
    mc = mcscf.CASSCF (mf, 4, 4)
    mc.fcisolver = csf_solver (mol, smult=1)
    mc.kernel ()
    #mo_coeff = mc.mo_coeff.copy ()
    print (mc.converged, mc.e_tot)
    las = LASSCFNoSymm (mf, (4,), ((2,2),), spin_sub=(1,))
    las.kernel (mo_coeff)

    mc.mo_coeff = mo_coeff.copy ()
    las.mo_coeff = mo_coeff.copy ()
    las.ci = [[mc.ci.copy ()]]

    nao, nmo = mo_coeff.shape
    ncore, ncas = mc.ncore, mc.ncas
    nocc = ncore + ncas
    eris = mc.ao2mo ()
    from pyscf.mcscf.newton_casscf import gen_g_hop, _pack_ci_get_H
    g_all_cas, g_update_cas, h_op_cas, hdiag_all_cas = gen_g_hop (mc, mo_coeff, mc.ci, eris)
    _pack_ci, _unpack_ci = _pack_ci_get_H (mc, mo_coeff, mc.ci)[-2:]
    nvar_orb_cas = np.count_nonzero (mc.uniq_var_indices (nmo, mc.ncore, mc.ncas, mc.frozen))
    nvar_tot_cas = g_all_cas.size
    def pack_cas (kappa, ci1):
        return np.append (mc.pack_uniq_var (kappa), _pack_ci (ci1))
    def unpack_cas (x):
        return mc.unpack_uniq_var (x[:nvar_orb_cas]), _unpack_ci (x[nvar_orb_cas:])

    ugg = las.get_ugg (las.mo_coeff, las.ci)
    h_op_las = las.get_hop (ugg=ugg)

    print ("Total # variables: {} CAS ; {} LAS".format (nvar_tot_cas, ugg.nvar_tot))
    print ("# orbital variables: {} CAS ; {} LAS".format (nvar_orb_cas, ugg.nvar_orb))

    gorb_cas, gci_cas = unpack_cas (g_all_cas)
    gorb_las, gci_las = ugg.unpack (h_op_las.get_grad ())
    gorb_cas /= 2.0 # Newton-CASSCF multiplies orb-grad terms by 2 so as to exponentiate kappa instead of kappa/2

    # For orb degrees of freedom, gcas = 2 glas and therefore 2 xcas = xlas
    print (" ")
    print ("Orbital gradient norms: {} CAS ; {} LAS".format (linalg.norm (gorb_cas), linalg.norm (gorb_las)))
    print ("Orbital gradient disagreement:", linalg.norm (gorb_cas-gorb_las))
    print ("CI gradient norms: {} CAS ; {} LAS".format (linalg.norm (gci_cas), linalg.norm (gci_las)))
    print ("CI gradient disagreement:", linalg.norm (gci_cas[0]-gci_las[0][0]))
                

    np.random.seed (0)
    x = np.random.rand (ugg.nvar_tot)
    xorb, xci = ugg.unpack (x)
    def examine_sector (sector, xorb_inp, xci_inp):
        xorb = np.zeros_like (xorb_inp)
        xci = [[np.zeros_like (xci_inp[0][0])]]
        ij = {'core': (0,ncore),
              'active': (ncore,nocc),
              'virtual': (nocc,nmo)}
        if sector.upper () == 'CI':
            xci[0][0] = xci_inp[0][0].copy ()
        else:
            bra, ket = sector.split ('-')
            i, j = ij[bra]
            k, l = ij[ket]
            xorb[i:j,k:l] = xorb_inp[i:j,k:l]
            xorb[k:l,i:j] = xorb_inp[k:l,i:j]

        # Compensate for PySCF's failure to intermediate-normalize
        cx = mc.ci.ravel ().dot (xci[0][0].ravel ())
        xci_cas = [xci[0][0] - (cx * mc.ci.ravel ())]
       
        x_las = ugg.pack (xorb, xci)
        x_cas = pack_cas (xorb, xci_cas) 
        hx_orb_las, hx_ci_las = ugg.unpack (h_op_las._matvec (x_las))
        hx_orb_cas, hx_ci_cas = unpack_cas (h_op_cas (x_cas))

        # Subtract the level shift that I put in there on purpose
        dhx_orb_las, dhx_ci_las = ugg.unpack (las.ah_level_shift * np.abs (x_las))
        hx_orb_las -= dhx_orb_las
        hx_ci_las[0][0] -= dhx_ci_las[0][0]

        hx_orb_cas /= 2.0
        if sector.upper () != 'CI':
            hx_ci_cas[0] /= 2.0 
            # g_orb (PySCF) = 2 * g_orb (LASSCF) ; hx_orb (PySCF) = 2 * hx_orb (LASSCF)
            # This means that x_orb (PySCF) = 0.5 * x_orb (LASSCF), which must have been worked into the
            # derivation of the (h_co x_o) sector of hx. Passing the same numbers into the newton_casscf
            # hx calculator will involve orbital distortions which are twice as intense as my hx 
            # hx calculator, and the newton_casscf CI sector of the hx will be twice as large

        # More intermediate normalization
        ci_norm = np.dot (hx_ci_las[0][0].ravel (), hx_ci_las[0][0].ravel ())
        chx_cas = mc.ci.ravel ().dot (hx_ci_cas[0].ravel ())
        hx_ci_cas[0] -= chx_cas * mc.ci.ravel ()

        print (" ")
        for osect in ('core-virtual', 'active-virtual', 'core-active'):
            bra, ket = osect.split ('-')
            i, j = ij[bra]
            k, l = ij[ket]
            print ("{} - {} Hessian sector".format (osect, sector))
            print ("Hx norms: {} CAS ; {} LAS".format (linalg.norm (hx_orb_cas[i:j,k:l]), linalg.norm (hx_orb_las[i:j,k:l])))
            print ("Hx disagreement:".format (osect, sector),
                linalg.norm (hx_orb_cas[i:j,k:l]-hx_orb_las[i:j,k:l]))
        print ("CI - {} Hessian sector".format (sector))
        print ("Hx norms: {} CAS ; {} LAS".format (linalg.norm (hx_ci_cas), linalg.norm (hx_ci_las)))
        print ("Hx disagreement:", linalg.norm (hx_ci_las[0][0]-hx_ci_cas[0]), np.dot (hx_ci_las[0][0].ravel (), hx_ci_cas[0].ravel ()) / ci_norm)
        chx_cas = mc.ci.ravel ().dot (hx_ci_cas[0].ravel ())
        chx_las = mc.ci.ravel ().dot (hx_ci_las[0][0].ravel ())
        print ("CI intermediate normalization check: {} CAS ; {} LAS".format (chx_cas, chx_las))


    examine_sector ('core-virtual', xorb, xci)
    examine_sector ('active-virtual', xorb, xci)
    examine_sector ('core-active', xorb, xci)
    examine_sector ('CI', xorb, xci)
    print ("\nNotes:")
    print ("1) Newton_casscf.py multiplies all orb grad terms by 2 and exponentiates by kappa as opposed to kappa/2. This is accounted for in the above.")
    print ("2) Newton_casscf.py has a bug in the H_cc component that breaks intermediate normalization, which almost never matters but which I've compensated for above anyway.")
    print ("3) My preconditioner is obviously wrong, but in such a way that it suppresses CI vector evolution, which means stuff still converges if the CI vector isn't super sensitive to the orbital rotations.")

    from mrh.tests.lasscf.c2h4n4_struct import structure as struct
    mo0 = np.loadtxt ('/home/herme068/gits/mrh/tests/lasscf/test_lasci_mo.dat')
    ci00 = np.loadtxt ('/home/herme068/gits/mrh/tests/lasscf/test_lasci_ci0.dat')
    ci01 = np.loadtxt ('/home/herme068/gits/mrh/tests/lasscf/test_lasci_ci1.dat')
    ci0 = None #[[ci00], [-ci01.T]]
    dr_nn = 3.0
    mol = struct (dr_nn, dr_nn, '6-31g', symmetry=False)
    mol.verbose = lib.logger.DEBUG
    mol.output = 'lasscf_o0_c2n4h4.log'
    mol.spin = 0 
    #mol.symmetry = 'Cs'
    mol.build ()
    mf = scf.RHF (mol).run ()
    mc = LASSCFNoSymm (mf, (4,4), ((4,0),(4,0)), spin_sub=(5,5))
    mc.ah_level_shift = 1e-4
    mo = mc.localize_init_guess ((list(range(3)),list(range(7,10))))
    tools.molden.from_mo (mol, 'localize_init_guess.molden', mo)
    mc.kernel (mo)
    tools.molden.from_mo (mol, 'c2h4n4_opt.molden', mc.mo_coeff)

