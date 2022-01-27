from pyscf.mcscf import newton_casscf
from pyscf.grad import rks as rks_grad
from pyscf.dft import gen_grid
from pyscf.lib import logger, pack_tril, current_memory, tag_array
from pyscf.grad import sacasscf
from pyscf.mcscf.casci import cas_natorb
from mrh.my_pyscf.mcpdft.otpd import get_ontop_pair_density, _grid_ao2mo
from mrh.my_pyscf.mcpdft.pdft_veff import _contract_vot_rho, _contract_ao_vao
from mrh.util.rdm import get_2CDM_from_2RDM
from functools import reduce
from scipy import linalg
import numpy as np
import time, gc
from pyscf.data import nist
from pyscf import lib
from mrh.my_pyscf.grad import mcpdft
from mrh.my_pyscf.grad import sipdft


def mcpdft_HellmanFeynman_dipole (mc, ot, veff1, veff2, mo_coeff=None, ci=None, atmlst=None, mf_grad=None, verbose=None, max_memory=None, auxbasis_response=False):
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci
    if mf_grad is None: mf_grad = mc._scf.nuc_grad_method()
    if mc.frozen is not None:
        raise NotImplementedError
    if max_memory is None: max_memory = mc.max_memory
    t0 = (logger.process_clock (), logger.perf_counter ())

    mol = mc.mol
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas
    nelecas = mc.nelecas

    mo_core = mo_coeff[:,:ncore]
    mo_cas = mo_coeff[:,ncore:nocc]
 
    casdm1 = mc.fcisolver.make_rdm1(ci, ncas, nelecas)
 
    dm_core = np.dot(mo_core, mo_core.T) * 2
    dm_cas = reduce(np.dot, (mo_cas, casdm1, mo_cas.T))
    dm = dm_core + dm_cas
 
    with mol.with_common_orig((0,0,0)):
        ao_dip = mol.intor_symmetric('int1e_r', comp=3)
    el_dip = np.einsum('xij,ij->x', ao_dip, dm).real
 
    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    nucl_dip = np.einsum('i,ix->x', charges, coords)
    cas_dip = nucl_dip - el_dip

    return cas_dip

class ElectricDipole (sipdft.Gradients):

    def kernel (self, level_shift=None, unit='Debye', **kwargs):
        ''' Cache the effective Hamiltonian terms so you don't have to calculate them twice '''
        state = kwargs['state'] if 'state' in kwargs else self.state
        if state is None:
            raise NotImplementedError ('Gradient of PDFT state-average energy')
        self.state = state # Not the best code hygiene maybe
        mo = kwargs['mo'] if 'mo' in kwargs else self.base.mo_coeff
        ci = kwargs['ci'] if 'ci' in kwargs else self.base.ci
        si = kwargs['si'] if 'si' in kwargs else self.base.si
        if isinstance (ci, np.ndarray): ci = [ci] # hack hack hack...
        kwargs['ci'] = ci

        nroots = self.nroots
        veff1 = []
        veff2 = []
        for ix in range (nroots):
            v1, v2 = self.base.get_pdft_veff (mo, ci, incl_coul=True,
                paaa_only=True, state=state)
            veff1.append (v1)
            veff2.append (v2)
        kwargs['veff1'], kwargs['veff2'] = veff1, veff2

        #kwargs['veff1'], kwargs['veff2'] = self.base.get_pdft_veff (mo, ci, incl_coul=True, paaa_only=True, state=state)
        cput0 = (logger.process_clock(), logger.perf_counter())
        log = lib.logger.new_logger(self, self.verbose)
        if 'atmlst' in kwargs:
            self.atmlst = kwargs['atmlst']

        if self.verbose >= lib.logger.WARN:
            self.check_sanity()
        if self.verbose >= lib.logger.INFO:
            self.dump_flags()
        
        d2f = self.base.sarot_objfn (ci=ci)[2]
        conv, Lvec, bvec, Aop, Adiag = self.solve_lagrange (level_shift=level_shift, **kwargs)
        self.debug_lagrange (Lvec, bvec, Aop, Adiag, d2f=d2f, **kwargs)
        #if not conv: raise RuntimeError ('Lagrange multiplier determination not converged!')
        cput1 = lib.logger.timer (self, 'Lagrange gradient multiplier solution', *cput0)

        ci_final = np.tensordot (si.T, np.stack (ci, axis=0), axes=1)
        kwargs['ci'] = ci_final
#        fcasscf = self.make_fcasscf (state)
#        fcasscf.mo_coeff = mo
#        fcasscf.ci = ci[state]
        ham_response = self.get_ham_response (**kwargs)
#        lib.logger.info(self, '--------------- %s gradient Hamiltonian response ---------------',
#                    self.base.__class__.__name__)
#        rhf_grad._write(self, self.mol, ham_response, self.atmlst)
#        lib.logger.info(self, '----------------------------------------------')
#        cput1 = lib.logger.timer (self, 'Lagrange gradient Hellmann-Feynman determination', *cput1)

        LdotJnuc = self.get_LdotJnuc (Lvec, **kwargs)
#        lib.logger.info(self, '--------------- %s gradient Lagrange response ---------------',
#                    self.base.__class__.__name__)
#        rhf_grad._write(self, self.mol, LdotJnuc, self.atmlst)
#        lib.logger.info(self, '----------------------------------------------')
#        cput1 = lib.logger.timer (self, 'Lagrange gradient Jacobian', *cput1)

        
        mol_dip = ham_response + LdotJnuc

        if unit.upper() == 'DEBYE':
            ham_response *= nist.AU2DEBYE
            mol_dip      *= nist.AU2DEBYE
            LdotJnuc     *= nist.AU2DEBYE
            log.note('CASSCF  Dipole moment(X, Y, Z, Debye): %8.5f, %8.5f, %8.5f', *ham_response)
            log.note('Lagrange Contribution(X, Y, Z, Debye): %8.5f, %8.5f, %8.5f', *LdotJnuc)
            log.note('MC-PDFT Dipole moment(X, Y, Z, Debye): %8.5f, %8.5f, %8.5f', *mol_dip)
        else:
            log.note('CASSCF  Dipole moment(X, Y, Z, A.U.): %8.5f, %8.5f, %8.5f', *ham_response)
            log.note('Lagrange Contribution(X, Y, Z, A.U.): %8.5f, %8.5f, %8.5f', *LdotJnuc)
            log.note('MC-PDFT Dipole moment(X, Y, Z, A.U.): %8.5f, %8.5f, %8.5f', *mol_dip)
        #print('ham_response = %f' %ham_response)
        #print('LdotJnuc = %f' %LdotJnuc)
        return mol_dip

    def get_ham_response (self, state=None, atmlst=None, verbose=None, mo=None, ci=None, eris=None, mf_grad=None, veff1=None, veff2=None, si=None, **kwargs):
        if state is None: state = self.state
        if atmlst is None: atmlst = self.atmlst
        if verbose is None: verbose = self.verbose
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if si is None: si = self.base.si
        if (veff1 is None) or (veff2 is None):
            assert (False), kwargs
            veff1, veff2 = self.base.get_pdft_veff (mo, ci[state], incl_coul=True, paaa_only=True)
        fcasscf = self.make_fcasscf (state)
        fcasscf.mo_coeff = mo
        #new_ci = np.tensordot(si.T, np.stack(ci,axis = 0), axes = 1)
        fcasscf.ci = ci[state]
        return mcpdft_HellmanFeynman_dipole (fcasscf, self.base.otfnal, veff1, veff2, mo_coeff=mo, ci=ci[state], atmlst=atmlst, mf_grad=mf_grad, verbose=verbose)

    def get_LdotJnuc (self, Lvec, state=None, atmlst=None, verbose=None, mo=None, ci=None, eris=None, mf_grad=None, si=None, **kwargs):
        if state is None: state = self.state
        if atmlst is None: atmlst = self.atmlst
        if verbose is None: verbose = self.verbose
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if si is None: si = self.base.si
        if eris is None and self.eris is None:
            eris = self.eris = self.base.ao2mo (mo)
        elif eris is None:
            eris = self.eris
        ncas = self.base.ncas
        nelecas = self.base.nelecas
        if getattr(self.base.fcisolver, 'gen_linkstr', None):
            linkstr  = self.base.fcisolver.gen_linkstr(ncas, nelecas, False)
        else:
            linkstr  = None
        mc = self.base
     #   mc = self.make_fcasscf (state)
     #   fcasscf = self.make_fcasscf (state)
     #   fcasscf.mo_coeff = mo
     #   fcasscf.ci = ci[state]


        # Just sum the weights now... Lorb can be implicitly summed
        # Lci may be in the csf basis
        Lorb, Lci, _ = self.unpack_uniq_var (Lvec)

        mo_coeff = mc.mo_coeff
        ci = mc.ci
        new_ci = np.tensordot(si.T, np.stack(ci,axis = 0), axes = 1)
    
        mol = mc.mol
        ncore = mc.ncore
        ncas = mc.ncas
        nocc = ncore + ncas
        nelecas = mc.nelecas
    
        mo_occ = mo_coeff[:,:nocc]
        mo_core = mo_coeff[:,:ncore]
        mo_cas = mo_coeff[:,ncore:nocc]
    

        # Orb part
        # MRH: new 'effective' MO coefficients including contraction from the Lagrange multipliers
        moL_coeff = np.dot (mo_coeff, Lorb)
        moL_core = moL_coeff[:,:ncore]
        moL_cas = moL_coeff[:,ncore:nocc]
    
        # MRH: these SHOULD be state-averaged! Use the actual sacasscf object!
        casdm1 = mc.fcisolver.make_rdm1(ci, ncas, nelecas)
    
        # MRH: new density matrix terms
        dmL_core = np.dot(moL_core, mo_core.T) * 2
        dmL_cas = reduce(np.dot, (moL_cas, casdm1, mo_cas.T))
        dmL_core += dmL_core.T
        dmL_cas += dmL_cas.T

        # MRH: TDMs + c.c. instead of RDMs; 06/30/2020: new interface in mcscf.addons makes this much more transparent
        # MRH 12/06/2021: When Aleks wrote this, he falsely assumed that I had implemented
        # trans_rdm1 as well as trans_rdm12 for the state-averaged case. Luckily there's a quick way
        # to fix this without pushing something to PySCF at the moment: just make both 1 and 2 and discard 2
        
        casdm1_transit, _ = mc.fcisolver.trans_rdm12 (Lci, ci, ncas, nelecas)
        #print('shape::: ', casdm1_transit)
        #casdm1_transit_vec = [i[0] for i in casdm_transit]
        #casdm1_transit = np.sum([weigths[i] * casdm1_transit_vec[i] for i in range(len(weights))])
        casdm1_transit += casdm1_transit.transpose (1,0)

        dm_cas_transit = reduce(np.dot, (mo_cas, casdm1_transit, mo_cas.T))

        dm = dmL_core + dmL_cas + dm_cas_transit

        with mol.with_common_orig((0,0,0)):
            ao_dip = mol.intor_symmetric('int1e_r', comp=3)
        mol_dip_L = -np.einsum('xij,ji->x', ao_dip, dm).real
        
#transition dipole moments
#        for i in range(len(weights)):
 #           for j in range(len(weights)):
  #              t_dm1 = mc.fcisolver.trans_rdm1(new_ci[i], new_ci[j], ncas, nelecas)
   #             orbcas = mo_coeff[:,ncore:ncore+ncas]
    #            t_dm1_ao = reduce(np.dot, (orbcas, t_dm1, orbcas.T))
     #           tot_mol[i][j] =  -np.einsum('xij,ji->x', ao_dip, t_dm1_ao).real



        return mol_dip_L


if __name__ == '__main__':
    from tabulate import tabulate
    from pyscf import gto, scf, mcscf
    from pyscf.lib import logger
    from mrh.my_pyscf import mcpdft
    from mrh.my_pyscf.fci import csf_solver
    from scipy import linalg
    import numpy as np
    #import os
    from pyscf.tools import molden



    def get_dipole_CMSPDFT(dist, norb, nel, mo, icharge, ispin, isym, cas_list, weights):
        out = "CO_"+basis_name+'_'+str(dist)
        mol = gto.M(atom="C   0.0              0.0              0.0; "         +
                         "O   0.0              0.0 "            + str(dist),
                    charge=icharge, spin=ispin, output=out+'.log', verbose=4, basis=my_basis)

        # HF
        mf = scf.RHF(mol).run()
        molden.from_mo(mol, out+'_hf'+'.molden', mf.mo_coeff)

        # CASSCF
        cas = mcscf.CASSCF(mf, norb, nel)
        cas.natorb = True
        # cas.fcisolver.conv_tol = 1e-8
        if mo is None:
            print('NONE ORBS')
            mo = mcscf.sort_mo(cas, mf.mo_coeff, cas_list)
        else:
            mo = mcscf.project_init_guess(cas, mo)

        e_cas = cas.kernel(mo)[0]
        cas.analyze()
        mo = cas.mo_coeff
        molden.from_mo(mol, out+'.molden', cas.mo_coeff)

        #MC-PDFT
        mc = mcpdft.CASSCF(mf, 'tPBE', norb, nel, grids_level=9)
        mc.fcisolver = csf_solver(mol, smult=ispin+1)
        mc.fcisolver.max_cycle = 200
        mc.max_cycle_macro = 200
        mc.fcisolver.wfnsym = isym

        e_pdft = mc.kernel(mo)[0]
        mc = mc.state_interaction(weights,'cms').run()

        dip_pdft = []
        dip_pdft_full = []

        for i in range(len(weights)):
            dip_pdft.append(ElectricDipole(mc).kernel(unit='Debye',state = i))   #TODO
            dip_pdft_full.append(np.linalg.norm(dip_pdft[i]))

        new_ci = np.tensordot(mc.si.T, np.stack(mc.ci,axis = 0), axes = 1)
        tot_dip = np.diag(dip_pdft_full)
        
        with mol.with_common_orig((0,0,0)):
            ao_dip = mol.intor_symmetric('int1e_r', comp=3)

        #transition dipole moments
        for i in range(len(weights)):
            for j in range(len(weights)):
                if i == j:
                    continue
                t_dm1 = mc.fcisolver.trans_rdm1(new_ci[i], new_ci[j], mc.ncas, mc.nelecas)
                orbcas = mc.mo_coeff[:,mc.ncore:mc.ncore+mc.ncas]
                t_dm1_ao = reduce(np.dot, (orbcas, t_dm1, orbcas.T))
                tot_dip[i][j] =  np.linalg.norm(-np.einsum('xij,ji->x', ao_dip, t_dm1_ao).real)

        print(tot_dip)

        x_a, y_a, z_a = [[dip[i] for dip in dip_pdft] for i in range(3)]  #reshape dip_pdft to components

        data = [np.concatenate(([i,dist,e_cas, e_pdft],[x_a[i],y_a[i],z_a[i],dip_pdft_full[i]]),axis=None) for i in range(len(weights))]
        #for i in range(len(weights)):
         #   data.append(np.concatenate(([i,dist,e_cas, e_pdft],[x_a[i],y_a[i],z_a[i],dip_pdft_full[i]]),axis=None))

        return data, mo


    def pdtabulate(df, line1, line2): return tabulate(
        df, headers=line1, tablefmt='psql', floatfmt=line2)

    def print_numerical_dipole():

        string = '\n\nNumerical reference:\n+----------+------------+--------------+--------------+-----------+------------------+\n|CMS State |   Distance |            X |            Y |         Z |   Dipole Numeric |\n|----------+------------+--------------+--------------+-----------+------------------|\n|  0       |    1.14193 |  9.26064e-06 | -0.00016335  |  0.169934 |         0.169934 |\n|  1       |    1.14193 |  8.9579e-06  | -0.000203217 | -0.269189 |         0.269189 |\n|  2       |    1.14193 |  8.58558e-07 | -8.95043e-05 |  1.74604  |         1.74604  |\n+----------+------------+--------------+--------------+-----------+------------------+\n'
        print(string)

    def run(norb, nel, mo, array, icharge, ispin, isym, cas_list, weights):

        data = []
        for i, dist in enumerate(array, start=0):
            data_tmp, _ = get_dipole_CMSPDFT(dist, norb, nel, mo, icharge, ispin, isym, cas_list, weights)
            data.append(data_tmp)
        
        data = [line for dat in data for line in dat]

        list = data

        # Final scan table
        line1 = ['CMS State','Distance', 'Energy CASSCF', 'Energy MC-PDFT', 'X', 'Y', 'Z',
                    'Dipole Analytic']
        line2 = (".0f",".5f", ".8f", ".8f", ".6f", ".6f", ".6f",
                    ".6f")
        print(pdtabulate(list, line1, line2))
        with open('CO_'+str(nel)+'x'+str(norb)+'_'+basis_name+'.txt', 'w') as f:
            f.write(pdtabulate(list, line1, line2))
        print_numerical_dipole()

    #MAIN
    array = np.array([1.1419269322867753])
    isym='A1'
    ispin=0
    icharge=0
    nel, norb  = 10, 10
    cas_list = [i for i in range(3,13)]
    ndata = len(array)
    list = [0]*ndata
    pdft = [0]*ndata
    basis_name = 'augccpvdz'
    my_basis = basis_name
    dm1 = mo = None
    weights = [1/2]*2

    run(norb, nel, mo, array, icharge, ispin, isym, cas_list, weights)
