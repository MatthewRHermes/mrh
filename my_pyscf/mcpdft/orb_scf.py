import numpy as np
from scipy import linalg
from pyscf import lib
from pyscf.mcscf import mc1step

def mc1step_gen_g_hop (mc, mo, u, casdm1, casdm2, eris):
    ''' Wrapper to mc1step.gen_g_hop for minimizing the PDFT energy
        instead of the CASSCF energy by varying orbitals '''
    ncore, ncas, nelecas = mc.ncore, mc.ncas, mc.nelecas
    nocc = ncore + ncas
    nao, nmo = mo.shape
    casdm1s = casdm1*0.5
    casdm1s = np.stack ([casdm1s, casdm1s], axis=0)
    veff1, veff2 = mc.get_pdft_veff (mo=mo, casdm1s=casdm1s, casdm2=casdm2,
        incl_coul=True)
    def get_hcore (mol=None):
        return mc._scf.get_hcore (mol) + veff1
    with lib.temporary_env (mc, get_hcore=get_hcore):
        g_orb, _, h_op, h_diag = mc1step.gen_g_hop (mc, mo, u, casdm1, casdm2,
            veff2)
    def gorb_update (u, ci_):
        # I have to reimplement this because of ci-dependence of veff1 and veff2
        # It's less complicated to write, but is is going to be pretty costly
        mo_ = np.dot(mo, u)
        veff1_, veff2_ = mc.get_pdft_veff (mo=mo_, ci=ci_, incl_coul=True,
            paaa_only=True)
        h1e_mo_ = ((mo_.T @ (mc._scf.get_hcore (mc.mol) + veff1_) @ mo_)
                   + veff2_.vhf_c)
        casdm1_, casdm2_ = mc.fcisolver.make_rdm12(ci_, ncas, nelecas)
        aapa = np.zeros ((ncas,ncas,nmo,ncas), dtype=h1e_mo_.dtype)
        vhf_a = np.zeros ((nmo,nmo), dtype=h1e_mo_.dtype)
        for i in range (nmo):
            jbuf = veff2_.ppaa[i]
            aapa[:,:,i,:] = jbuf[ncore:nocc,:,:]
            vhf_a[i] = np.tensordot (jbuf, casdm1_, axes=2)
        vhf_a *= 0.5
        # for this potential, vj = vk: vj - vk/2 = vj - vj/2 = vj/2
        g = np.zeros ((nmo, nmo))
        g[:,:ncore] = (h1e_mo_[:,:ncore] + vhf_a[:,:ncore]) * 2
        g[:,ncore:nocc] = h1e_mo_[:,ncore:nocc] @ casdm1_
        g[:,ncore:nocc] += np.einsum('uviw,vuwt->it', aapa, casdm2_)
        return mc.pack_uniq_var(g-g.T)
    return g_orb, gorb_update, h_op, h_diag

def mc1step_update_jk_in_ah (mc, mo_coeff, x1, casdm1, veff2):
    ''' Wrapper to mc1step.update_jk_in_ah for minimizing the PDFT energy
        instead of the CASSCF energy by varying orbitals '''
    ncore, ncas = mc.ncore, mc.ncas
    nocc = ncore + ncas
    nao, nmo = mo_coeff.shape

    # Density matrices
    dm1 = 2 * np.eye (nmo)
    dm1[nocc:,nocc:] = 0.0
    dm1[ncore:nocc,ncore:nocc] = casdm1
    ddm1 = mo_coeff @ x1 @ dm1 @ mo_coeff.conj ().T

    # Coulomb 
    ddm1 += ddm1.T
    dvj = mo_coeff.conj ().T @ mc._scf.get_j (dm=ddm1) @ mo_coeff
    dg = dm1 @ dvj # convention of mc1step: density matrix index first

    # TODO: OT; requires more thought

    # comply with return signature
    dg_a = dg[ncore:nocc]
    dg_c = dg[:ncore,ncore:]
    return dg_a, dg_c    


if __name__ == '__main__':
    from pyscf import gto, scf
    from mrh.my_pyscf import mcpdft
    mol = gto.M (atom = 'Li 0 0 0; Li 1.2 0 0', basis = '6-31g',
        verbose=lib.logger.DEBUG, output='orb_scf.log')
    mf = scf.RHF (mol).run ()
    mc = mcpdft.CASSCF (mf, 'tPBE', 2, 2, grids_level=9).run ()
    print ("Ordinary Li2 tPBE energy:",mc.e_tot)

    nao, nmo = mc.mo_coeff.shape
    ncore, ncas, nelecas = mc.ncore, mc.ncas, mc.nelecas
    nocc = ncore+ncas

    casdm1, casdm2 = mc.fcisolver.make_rdm12 (mc.ci, ncas, nelecas)
    g_orb, gorb_update, h_op, h_diag = mc1step_gen_g_hop (mc,
        mc.mo_coeff, 1, casdm1, casdm2, None)

    print ("g_orb:", linalg.norm (g_orb))
    print ("gorb_update (1,mc.ci):", linalg.norm (gorb_update (1, mc.ci)))
    print ("h_op(0):", linalg.norm (h_op (np.zeros (60))))
    print ("h_diag:", linalg.norm (h_diag))

    x0 = -g_orb/h_diag
    u0 = mc.update_rotate_matrix (x0)
    print ("\nx0 = -g_orb/h_diag; u0 = expm (x0)")
    print ("h_op(x0):", linalg.norm (h_op(x0)))
    print ("gorb_update (u0,mc.ci):", linalg.norm (gorb_update (u0, mc.ci)))

    mc.mo_coeff = np.dot (mc.mo_coeff, u0)
    e_tot, e_ot = mc.energy_tot ()
    print ("Li2 tPBE energy after rotating orbitals:", e_tot)

