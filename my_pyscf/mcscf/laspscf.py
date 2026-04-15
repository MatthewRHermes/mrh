import numpy as np
from scipy import linalg
from mrh.util.la import matrix_svd_control_options
from mrh.my_pyscf.mcscf import lasscf_sync_o0, _DFLASCI
from mrh.my_pyscf.mcscf import lasscf_guess
from pyscf import gto, scf, symm, lib
from pyscf.mcscf import mc1step
from pyscf.lo import orth
from pyscf.lib import tag_array, with_doc, logger
from functools import partial

localize_init_guess=lasscf_guess._localize # backwards compatibility

class LASPSCF_UnitaryGroupGenerators (lasscf_sync_o0.LASSCF_UnitaryGroupGenerators):

    def _init_orb (self, las, mo_coeff, ci):
        lasscf_sync_o0.LASSCF_UnitaryGroupGenerators._init_nonfrozen_orb (self, las)
        idx = self.nfrz_orb_idx.copy ()
        ncore, nocc = las.ncore, las.ncore + las.ncas
        idx[ncore:nocc,:ncore] = False # no inactive -> active
        idx[nocc:,ncore:nocc] = False # no active -> virtual
        # No external rotations of active orbitals
        self.uniq_orb_idx = idx

class LASPSCFSymm_UnitaryGroupGenerators (LASPSCF_UnitaryGroupGenerators):
    __init__ = lasscf_sync_o0.LASSCFSymm_UnitaryGroupGenerators.__init__
    _init_ci = lasscf_sync_o0.LASSCFSymm_UnitaryGroupGenerators._init_ci
    def _init_orb (self, las, mo_coeff, ci, orbsym):
        LASPSCF_UnitaryGroupGenerators._init_orb (self, las, mo_coeff, ci)
        self.symm_forbid = (orbsym[:,None] ^ orbsym[None,:]).astype (np.bool_)
        self.uniq_orb_idx[self.symm_forbid] = False
        self.nfrz_orb_idx[self.symm_forbid] = False

class LASPSCF_HessianOperator (lasscf_sync_o0.LASSCF_HessianOperator):
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

    _init_eri_ = lasscf_sync_o0._init_df_

    def get_veff (self, dm1s_mo=None):
        '''THIS FUNCTION IS OVERWRITTEN WITH A CALL TO LAS.GET_VEFF IN THE LASSCF_O0 CLASS. IT IS
        ONLY RELEVANT TO THE "LASSCF" STEP OF THE OLDER, DEPRECATED, DMET-BASED ALGORITHM.

        Compute the effective potential from a 1-RDM in the MO basis (presumptively the first-order
        effective 1-RDM which is proportional to a step vector in MO and CI rotation coordinates).
        If density fitting is used, the effective potential is approximate: it omits the
        unoccupied-unoccupied lower-diagonal block.

        Kwargs:
            dm1s_mo : ndarray of shape (2,nmo,nmo)
                Contains spin-separated 1-RDM

        Returns:
            veff_mo : ndarray of shape (nmo,nmo)
                Spin-symmetric effective potential in the MO basis
        '''

        mo = self.mo_coeff
        moH = mo.conjugate ().T
        nmo = mo.shape[-1]
        dm1_mo = dm1s_mo.sum (0)
        if self.las.use_gpu or (getattr(self, 'bPpj', None) is None):
            dm1_ao=np.dot(mo,np.dot(dm1_mo,moH))
            veff_ao=np.squeeze(self.las.get_veff(dm=dm1_ao))
            return np.dot(moH,np.dot(veff_ao,mo))
        ncore, nocc, ncas = self.ncore, self.nocc, self.ncas
        # vj
        t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
        veff_mo = np.zeros_like (dm1_mo) 
        dm1_rect = dm1_mo + dm1_mo.T
        dm1_rect[ncore:nocc,ncore:nocc] /= 2
        dm1_rect = dm1_rect[:,:nocc]
        rho = np.tensordot (self.bPpj, dm1_rect, axes=2)
        vj_pj = np.tensordot (rho, self.bPpj, axes=((0),(0)))
        t1 = lib.logger.timer (self.las, 'vj_mo in microcycle', *t0)
        dm_bj = dm1_mo[ncore:,:nocc]
        vPpj = np.ascontiguousarray (self.las.cderi_ao2mo (mo, mo[:,ncore:]@dm_bj, compact=False))
        # Don't ask my why this is faster than doing the two degrees of freedom separately...
        t1 = lib.logger.timer (self.las, 'vk_mo vPpj in microcycle', *t1)
        
        # vk (aa|ii), (uv|xy), (ua|iv), (au|vi)
        vPbj = vPpj[:,ncore:,:] #np.dot (self.bPpq[:,ncore:,ncore:], dm_ai)
        vk_bj = np.tensordot (vPbj, self.bPpj[:,:nocc,:], axes=((0,2),(0,1)))
        t1 = lib.logger.timer (self.las, 'vk_mo (bb|jj) in microcycle', *t1)
        # vk (ai|ai), (ui|av)
        dm_ai = dm1_mo[nocc:,:ncore]
        vPji = vPpj[:,:nocc,:ncore] #np.dot (self.bPpq[:,:nocc, nocc:], dm_ai)
        # I think this works only because there is no dm_ui in this case, so I've eliminated all
        # the dm_uv by choosing this range
        bPbi = self.bPpj[:,ncore:,:ncore]
        vk_bj += np.tensordot (bPbi, vPji, axes=((0,2),(0,2)))
        t1 = lib.logger.timer (self.las, 'vk_mo (bi|aj) in microcycle', *t1)
        t0 = lib.logger.timer (self.las, 'vj and vk mo', *t0)

        # veff
        vj_bj = vj_pj[ncore:,:]
        veff_mo[ncore:,:nocc] = vj_bj - 0.5*vk_bj
        veff_mo[:nocc,ncore:] = veff_mo[ncore:,:nocc].T
        #vj_ai = vj_bj[ncas:,:ncore]
        #vk_ai = vk_bj[ncas:,:ncore]
        #veff_mo[ncore:,:nocc] = vj_bj
        #veff_mo[:ncore,nocc:] = vj_ai.T
        #veff_mo[ncore:,:nocc] -= vk_bj/2
        #veff_mo[:ncore,nocc:] -= vk_ai.T/2
        return veff_mo

    def split_veff (self, veff_mo, dm1s_mo):
        # This function seems orphaned? Is it used anywhere?
        veff_c = veff_mo.copy ()
        ncore = self.ncore
        nocc = self.nocc
        dm1s_cas = dm1s_mo[:,ncore:nocc,ncore:nocc]
        sdm = dm1s_cas[0] - dm1s_cas[1]
        vk_aa = -np.tensordot (self.eri_cas, sdm, axes=((1,2),(0,1))) / 2
        veff_s = np.zeros_like (veff_c)
        veff_s[ncore:nocc, ncore:nocc] = vk_aa
        veffa = veff_c + veff_s
        veffb = veff_c - veff_s
        return np.stack ([veffa, veffb], axis=0)

    orbital_response = lasscf_sync_o0.LASSCF_HessianOperator.orbital_response_1cum

    def _update_h2eff_sub (self, mo1, umat, h2eff_sub):
        ncore, ncas, nocc, nmo = self.ncore, self.ncas, self.nocc, self.nmo
        ucas = umat[ncore:nocc, ncore:nocc]
        bmPu = None
        if hasattr (h2eff_sub, 'bmPu'):bmPu = h2eff_sub.bmPu
        h2eff_sub = h2eff_sub.reshape (nmo*ncas, ncas*(ncas+1)//2)
        h2eff_sub = lib.numpy_helper.unpack_tril (h2eff_sub)
        h2eff_sub = h2eff_sub.reshape (nmo, ncas, ncas, ncas)
        h2eff_sub = np.tensordot (ucas, h2eff_sub, axes=((0),(1))) # bpaa
        h2eff_sub = np.tensordot (umat, h2eff_sub, axes=((0),(1))) # qbaa
        h2eff_sub = np.tensordot (h2eff_sub, ucas, axes=((2),(0))) # qbab
        h2eff_sub = np.tensordot (h2eff_sub, ucas, axes=((2),(0))) # qbbb
        ix_i, ix_j = np.tril_indices (ncas)
        h2eff_sub = h2eff_sub.reshape (nmo, ncas, ncas*ncas)
        h2eff_sub = h2eff_sub[:,:,(ix_i*ncas)+ix_j]
        h2eff_sub = h2eff_sub.reshape (nmo, -1)
        if bmPu is not None:
            bmPu = np.dot (bmPu, ucas)
            h2eff_sub = lib.tag_array (h2eff_sub, bmPu = bmPu)
        return h2eff_sub

    def _update_h2eff_sub (self, mo1, umat, h2eff_sub):
        return self.las.ao2mo (mo1)

    def _get_Horb_diag_presymm (self):
        fock = np.stack ([np.diag (h) for h in list (self.h1s)], axis=0)
        num = np.stack ([np.diag (d) for d in list (self.dm1s)], axis=0)
        Horb_diag = sum ([np.multiply.outer (f,n) for f,n in zip (fock, num)])
        Horb_diag -= np.diag (self.fock1)[None,:]
        # This is where I stop unless I want to add the split-c and split-x terms
        # Split-c and split-x, for inactive-external rotations, requires I calculate a bunch
        # of extra eris (g^aa_ii, g^ai_ai)
        return Horb_diag

class LASPSCFNoSymm (lasscf_sync_o0.LASSCFNoSymm):
    _ugg = LASPSCF_UnitaryGroupGenerators
    _hop = LASPSCF_HessianOperator
    def dump_flags (self, verbose=None, _method_name='LASPSCF'):
        lasscf_sync_o0.LASSCFNoSymm.dump_flags (self, verbose=verbose, _method_name=_method_name)
    
class LASPSCFSymm (lasscf_sync_o0.LASSCFSymm):
    _ugg = LASPSCFSymm_UnitaryGroupGenerators    
    _hop = LASPSCF_HessianOperator
    dump_flags = LASPSCFNoSymm.dump_flags

def LASPSCF (mf_or_mol, ncas_sub, nelecas_sub, **kwargs):
    # try grabbing gpu handle from mf_or_mol instead of additional argument
    use_gpu = kwargs.get('use_gpu', None)
    
    if isinstance(mf_or_mol, gto.Mole):
        mf = scf.RHF(mf_or_mol)
    elif isinstance (mf_or_mol, scf.hf.SCF):
        mf = mf_or_mol
    else:
        raise RuntimeError ("LASSCF constructor requires molecule or SCF instance")
    if mf.mol.symmetry: 
        las = LASPSCFSymm (mf, ncas_sub, nelecas_sub, **kwargs)
    else:
        las = LASPSCFNoSymm (mf, ncas_sub, nelecas_sub, **kwargs)
    if getattr (mf, 'with_df', None):
        las = lasscf_sync_o0.density_fit (las, with_df = mf.with_df)
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
    mol.output = 'lasscf_sync_o0.log'
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
    mo0 = np.loadtxt ('/home/herme068/gits/mrh/tests/lasscf/test_lasscf_sync_o0_mo.dat')
    ci00 = np.loadtxt ('/home/herme068/gits/mrh/tests/lasscf/test_lasscf_sync_o0_ci0.dat')
    ci01 = np.loadtxt ('/home/herme068/gits/mrh/tests/lasscf/test_lasscf_sync_o0_ci1.dat')
    ci0 = None #[[ci00], [-ci01.T]]
    dr_nn = 3.0
    mol = struct (dr_nn, dr_nn, '6-31g', symmetry=False)
    mol.verbose = lib.logger.DEBUG
    mol.output = 'lasscf_sync_o0_c2n4h4.log'
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

