from pyscf.lib import logger
from pyscf.mcscf import mc1step
import numpy as np
from pyscf.data import nist
from pyscf import lib
from mrh.my_pyscf.prop.dip_moment import mspdft
from mrh.my_pyscf.grad.mspdft import mspdft_heff_response
from mrh.my_pyscf.grad.mspdft import _unpack_state

class TransitionDipole (mspdft.ElectricDipole):

    def convert_dipole (self, ham_response, LdotJnuc, mol_dip, unit='Debye'):
        val = np.linalg.norm(mol_dip)
        i   = self.state[0]
        j   = self.state[1]
        dif = abs(self.e_states[i]-self.e_states[j]) 
        osc = 2/3*dif*val**2
        if unit.upper() == 'DEBYE':
            ham_response *= nist.AU2DEBYE
            LdotJnuc     *= nist.AU2DEBYE
            mol_dip      *= nist.AU2DEBYE
        log = lib.logger.new_logger(self, self.verbose)
        log.note('CMS-PDFT TDM <{}|mu|{}>          {:>10} {:>10} {:>10}'.format(i,j,'X','Y','Z'))
        log.note('Hamiltonian Contribution (%s) : %9.5f, %9.5f, %9.5f', unit, *ham_response)
        log.note('Lagrange Contribution    (%s) : %9.5f, %9.5f, %9.5f', unit, *LdotJnuc)
        log.note('Transition Dipole Moment (%s) : %9.5f, %9.5f, %9.5f', unit, *mol_dip)
        log.note('Oscillator strength  : %9.5f', osc)
        return mol_dip

    def get_bra_ket(self, state, si):
        si_bra = si[:,state[1]]
        si_ket = si[:,state[0]]
        return si_bra, si_ket

    def get_ham_response (self, state=None, atmlst=None, verbose=None, mo=None,
                    ci=None, eris=None, si=None, **kwargs):
        if state is None: state = self.state
        if mo is None: mo = self.base.mo_coeff
        if ci is None: ci = self.base.ci
        if si is None: si = self.base.si

        fcasscf = self.make_fcasscf (state)
        fcasscf.mo_coeff = mo
        fcasscf.ci = ci
        elec_term = self.sipdft_HellmanFeynman_dipole (fcasscf, state=state, mo_coeff=mo, ci=ci, si=si)
        return elec_term