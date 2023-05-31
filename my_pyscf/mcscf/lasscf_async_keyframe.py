
class LASKeyframe (object):
    '''Shallow struct for various intermediates. DON'T put complicated code in here Matt!!!'''

    def __init__(self, las, mo_coeff, ci):
        self.las = las
        self.mo_coeff = mo_coeff
        self.ci = ci
        self._dm1s = self._veff = self._fock1 = self._h2eff_sub = None

    @property
    def dm1s (self):
        if self._dm1s is None:
            self._dm1s = self.las.make_rdm1s (mo_coeff=self.mo_coeff, ci=self.ci)
        return self._dm1s

    @property
    def veff (self):
        if self._veff is None:
            self._veff = self.las.get_veff (dm1s=self.dm1s, spin_sep=True)
        return self._veff

    @property
    def fock1 (self):
        if self._fock1 is None:
            self._fock1 = self.las.get_grad_orb (
                mo_coeff=self.mo_coeff, ci=self.ci, h2eff_sub=self.h2eff_sub, veff=self.veff,
                dm1s=self.dm1s)
        return self._fock1

    @property
    def h2eff_sub (self):
        if self._h2eff_sub is None:
            self._h2eff_sub = self.las.get_h2eff (self.mo_coeff)
        return self._h2eff_sub

    def copy (self):
        ''' MO coefficients deepcopy; CI vectors shallow copy. Everything else, drop. '''
        mo1 = self.mo_coeff.copy ()
        ci1_fr = []
        ci0_fr = self.ci
        for ci0_r in ci0_fr:
            ci1_r = []
            for ci0 in ci0_r:
                ci1 = ci0.view ()
                ci1_r.append (ci1)
            ci1_fr.append (ci1_r)
        return LASKeyframe (self.las, mo1, ci1_fr)
