from pyscf.dft.rks import _dft_common_init_
from mrh.my_pyscf.mcpdft import otfnal

def kernel (fnal, dm, max_memory=None, hermi=1):
    if max_memory is None: max_memory = fnal.max_memory
    s0 = fnal.get_ovlp ()
    dm_dot_dm = 0.5 * dm @ s0 @ dm
    dma = 1.5 * dm - dm_dot_dm
    dmb = dm_dot_dm - 0.5 * dm
    dms = np.stack ((dma,dmb), axis=0)
    # TODO: tag dms with MO information for speedup
    ni, xctype, dens_deriv = fnal._numint, fnal.xctype, fnal.dens_deriv

    Exc = 0.0
    make_rho = ni._gen_rho_evaluator (fnal.mol, dms, hermi)
    t0 = (time.clock (), time.time ())
    for ao, mask, weight, coords in ni.block_loop (fnal.mol, fnal.grids, norbs_ao, dens_deriv, max_memory):
        rho_eff = np.stack ((make_rho (spin, ao, mask, xctype) for spin in range (2)), axis=0)
        t0 = logger.timer (fnal, 'effective densities', *t0)
        Exc += fnal.get_E_xc (rho_eff, weight)
        t0 = logger.timer (fnal, 'exchange-correlation energy', *t0)
    return Exc

def _get_E_xc (fnal, rho_eff, weight):
    dexc_ddens  = fnal._numint.eval_xc (fnal.xc, (rho_eff[0,:,:], rho_eff[1,:,:]), spin=1, relativity=0, deriv=0, verbose=fnal.verbose)[0]
    rho_eff = rho_eff[:,0,:].sum (0)
    rho_eff *= weight
    dexc_ddens *= rho_eff

    if fnal.verbose >= logger.DEBUG:
        nelec = rho_eff.sum ()
        logger.debug (fnal, 'Total number of electrons in (this chunk of) the translated density = %s', nelec)

    return dexc_ddens.sum ()

class unpxcfnal (otfnal.otfnal)

    def __init__(self, mol, xc='LDA,WVN', grids_level=None):
        self.mol = mol
        self.verbose = mol.verbose
        self.stdout = mol.stdout
        self.max_memory = mol.max_memory
        _dft_common_init_(self, xc=xc)
        if grids_level is not None: self.grids.level = grids_level

    # Fix some names
    @property
    def otxc (self):
        return self.xc
    @property
    def get_E_ot (self, *args, **kwargs):
        return self.get_E_xc (*args, **kwargs)
    @property
    def get_dEot_drho (self, *args, **kwargs):
        return self.get_dExc_drho (*args, **kwargs)

    def get_ovlp (self, mol=None):
        if mol is None: mol = self.mol
        return mol.intor_symmetric('int1e_ovlp')

    get_E_xc = _get_E_xc
