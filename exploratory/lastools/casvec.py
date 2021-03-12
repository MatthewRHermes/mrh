import numpy as np

# subscripts like _frns tell you the structure of a complicated multidimensional array
# f : Fragments
# r : state (i.e., Root)
# n : hilbert-space sector for Number of particles
# s : Spin

class SubspaceEngine (object):
    def __init__(self, las):
        self.las = las
        self.ci = las.las2cas_civec ()
        self.nroots = nroots = las.nroots
        self.nfrags = nfrags = len (las.ncas_sub)
        self.norb_f = norb_f = np.asarray (las.ncas_sub)
        self.norb = norb = norb_f.sum ()
        self.nelec_frs = nelec_frs = np.zeros ((nfrags, nroots, 2), dtype=np.int32)
        for ifrag, (fcibox, nelec) in enumerate (zip (las.fciboxes, las.nelecas_sub)):
            for istate, fcisolver in enumerate (fcibox.fcisolvers):
                nelec_frs[ifrag,istate,:] = np.array (fcibox._get_nelec (solver, nelec))
        self.nelec_rs = nelec_rs = nelec_frs.sum (0)
        self.nmin_frs = nmin_frs = np.zeros ((nfrags, nroots, 2), dtype=np.int32)
        self.nmin_frs = nmin_frs = np.maximum (nmin_frs, nelec_rs[None,:,:] + norb_f[:,None,None] - norb)
        self.nmax_frs = nmax_frs = np.minimum (norb_f[:,None,None], nelec_rs[None,:,:])
        self.nmin_fs = np.amin (nmin_frs, axis=1)
        self.nmax_fs = np.amax (nmax_frs, axis=1)
        assert (np.all (self.nmax_frs >= self.nmin_frs))
        assert (np.all (self.nmax_fs >= self.nmin_fs))

