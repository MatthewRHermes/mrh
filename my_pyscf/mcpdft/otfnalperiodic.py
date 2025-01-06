from pyscf.mcpdft.otfnal import otfnal

import numpy as np
from pyscf.lib import logger
from pyscf.mcpdft import _dms
from pyscf.mcpdft.otpd import get_ontop_pair_density
from pyscf import __config__

def redefine_fnal(original_class, new_parent):
    # Create the new class
    class transfnal(new_parent):
        pass

    # Copy over attributes and methods
    for attr_name, attr_value in original_class.__dict__.items():
        if not attr_name.startswith("__"):
            setattr(transfnal, attr_name, attr_value)
    
    transfnal.eval_ot = original_class.eval_ot
    return transfnal

redefine_transfnal = redefine_fnal
redefine_ftransfnal = redefine_fnal

class otfnalperiodic(otfnal):
    '''
    Child class to define the otfnal class for periodic systems (Only for 1x1x1 kpts)
    '''

    def energy_ot (ot, casdm1s, casdm2, mo_coeff, ncore, max_memory=2000, hermi=1):
        '''
        See the docstring of pyscf/mcpdft/otfnal.energy_ot for more information.
        '''

        E_ot = 0.0
        ni = ot._numint
        xctype =  ot.xctype

        if xctype=='HF': 
            return E_ot
        
        dens_deriv = ot.dens_deriv

        nao = mo_coeff.shape[0]
        ncas = casdm2.shape[0]
        cascm2 = _dms.dm2_cumulant (casdm2, casdm1s)
        
        dm1s = _dms.casdm1s_to_dm1s (ot, casdm1s, mo_coeff=mo_coeff, ncore=ncore,
                                    ncas=ncas)
        mo_cas = mo_coeff[:,ncore:][:,:ncas]
        t0 = (logger.process_clock (), logger.perf_counter ())
        make_rho = tuple (ni._gen_rho_evaluator (ot.mol, dm1s[i,:,:], hermi) for
            i in range(2))
        
        for ao_k1, ao_k2, mask, weight, _ \
            in ni.block_loop(ot.mol, ot.grids, nao, deriv=dens_deriv, kpt=None, max_memory=max_memory):
            '''
            ao_k1 and ao_k2 are the block of AO integrals for the given k-point. They
            are the same for supercell(1x1x1) calculations.
            '''

            rho = np.asarray ([m[0] (0, ao_k1, mask, xctype) for m in make_rho])
            t0 = logger.timer (ot, 'untransformed density', *t0)
            Pi = get_ontop_pair_density (ot, rho, ao_k1, cascm2, mo_cas,
                dens_deriv, mask)
            t0 = logger.timer (ot, 'on-top pair density calculation', *t0)
            if rho.ndim == 2:
                rho = np.expand_dims (rho, 1)
                Pi = np.expand_dims (Pi, 0)
            E_ot += ot.eval_ot (rho, Pi, dderiv=0, weights=weight)[0].dot (weight)
            t0 = logger.timer (ot, 'on-top energy calculation', *t0)

        return E_ot
    
    energy_ot.__doc__ = otfnal.energy_ot.__doc__

    def reset(self, mol=None):
        '''
        Discard cached grid data and optionally update the cell object.
        I am not changing the input parameter so that it is compatible with the current
        MCPDFT code.
        '''
        if mol is not None:
            self.mol = mol
        # A hack to reset the grids for the new cell object.
        self.grids.reset (mol) 
    
def _get_transfnal (mol, otxc):
    '''
    This is wrapper function to get the appropriate fnal class for the given cell object
    '''
    from pyscf.mcpdft.otfnal import get_transfnal
    fnal_class = get_transfnal (mol, otxc)
    fnal_class_type = fnal_class.__class__.__name__
    
    assert isinstance(otxc, str), "The otxc should be a string"
    xc_base = fnal_class.otxc

    # Remember to write a sanity check that if this density_fit is not
    # GDF then raise an error.

    from pyscf.pbc import dft
    ks = dft.RKS(mol).density_fit()
    
    if fnal_class_type == 'transfnal':
        from pyscf.mcpdft.otfnal import transfnal
        xc_base = xc_base[1:]
        ks.xc = xc_base
        org_transfnal = transfnal(ks)
        new_func_class = redefine_transfnal (org_transfnal, otfnalperiodic)
        del org_transfnal

    elif fnal_class_type == 'ftransfnal':
        from pyscf.mcpdft.otfnal import ftransfnal
        xc_base = xc_base[1:]
        ks.xc = xc_base
        org_ftransfnal = ftransfnal(ks)
        new_func_class = redefine_ftransfnal (org_ftransfnal, otfnalperiodic)
        del org_transfnal
    else:
        raise ValueError ("The fnal class is not recognized")
    
    ks.xc = xc_base

    return new_func_class(ks)



            