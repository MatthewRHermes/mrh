'''
Atomic valence active orbitals (AVAS)
Ref. J. Chem. Theory Comput. 2017, 13, 4063−4078

Here, I have adapting the AVAS algorithm for the selection of active space 
with the PBC MCSCF. Probably with the k-point sampling as well!.
'''

import numpy as np
import scipy.linalg
from functools import reduce
from pyscf import lib
from pyscf.pbc import gto, scf
from pyscf import __config__
from pyscf.lib import logger

THRESHOLD = getattr(__config__, 'mcscf_avas_threshold', 0.2)
MINAO = getattr(__config__, 'mcscf_avas_minao', 'minao')
WITH_IAO = getattr(__config__, 'mcscf_avas_with_iao', False)
OPENSHELL_OPTION = getattr(__config__, 'mcscf_avas_openshell_option', 2)
CANONICALIZE = getattr(__config__, 'mcscf_avas_canonicalize', True)


def kernel(mf, aolabels, threshold=THRESHOLD, minao=MINAO, with_iao=WITH_IAO,
           openshell_option=OPENSHELL_OPTION, canonicalize=CANONICALIZE,
           ncore=0, verbose=None):
    '''AVAS method to construct mcscf active space.
    Args:
        mf : an :class:`SCF` object

        aolabels : string or a list of strings
            AO labels for AO active space

    Kwargs:
        threshold : float
            Tructing threshold of the AO-projector above which AOs are kept in
            the active space.
        minao : str
            A reference AOs for AVAS.
        with_iao : bool
            Whether to use IAO localization to construct the reference active AOs.
        openshell_option : int
            How to handle singly-occupied orbitals in the active space. The
            singly-occupied orbitals are projected as part of alpha orbitals
            if openshell_option=2, or completely kept in active space if
            openshell_option=3.  See Section III.E option 2 or 3 of the
            reference paper for more details.
        canonicalize : bool
            Orbitals defined in AVAS method are local orbitals.  Symmetrizing
            the core, active and virtual space.
        ncore : integer
            Number of core orbitals to be excluded from the AVAS method.

    Returns:
        active-space-size, #-active-electrons, orbital-initial-guess-for-CASCI/CASSCF
    '''

    # Detect whether the underline object is cell or mol
    # if it is mol, then point to molecular AVAS call.

    if hasattr(mf, 'cell'):
        mol_or_cell = mf.cell
    elif hasattr(mf, 'mol'):
        mol_or_cell = mf.mol
    else:
        raise RuntimeError('The input mf must be a SCF object with mol or cell')
    
    if getattr(mol_or_cell, 'pbc_intor', None):
        avas_obj = pbcAVAS(mf, aolabels, threshold, minao, with_iao,
                            openshell_option, canonicalize, ncore, verbose)
    else:
        avas_obj = molAVAS(mf, aolabels, threshold, minao, with_iao,
                            openshell_option, canonicalize, ncore, verbose)
    return avas_obj.kernel()

# For Gamma point
def _kernelGamma(avas_obj):
    log = logger.new_logger(avas_obj)
    log.info('\n** AVAS **')

    mf = avas_obj._scf
    cell = mf.cell
    nkpts = len(mf.kpts)

    assert nkpts == 1, 'For more than one kpts, use the other _kernel function'

    assert avas_obj.openshell_option != 1

    if isinstance(mf, scf.uhf.UHF):
        log.note('UHF/UKS object is found.  AVAS takes alpha orbitals only')
        mo_coeff = mf.mo_coeff[0]
        mo_occ = mf.mo_occ[0]
        mo_energy = mf.mo_energy[0]
    else:
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ
        mo_energy = mf.mo_energy

    ncore = avas_obj.ncore
    nocc = np.count_nonzero(mo_occ != 0)
    ovlp = mf.get_ovlp()
    log.info('  Total number of HF MOs  is equal to    %d' ,mo_coeff.shape[1])
    log.info('  Number of occupied HF MOs is equal to  %d', nocc)

    pcell = cell.copy()
    pcell.atom = cell._atom
    pcell.a = cell.a
    pcell.unit = 'B'
    pcell.symmetry = False
    pcell.basis = avas_obj.minao
    pcell.build(False, False)

    baslst = pcell.search_ao_label(avas_obj.aolabels)
    log.info('reference AO indices for %s %s:\n %s',
             avas_obj.minao, avas_obj.aolabels, baslst)

    if avas_obj.with_iao:
        raise NotImplemented
        # from pyscf.lo import iao
        # c = iao.iao(cell, mo_coeff[:,ncore:nocc], avas_obj.minao)[:,baslst]
        # s2 = reduce(np.dot, (c.T, ovlp, c))
        # s21 = reduce(np.dot, (c.T, ovlp, mo_coeff[:, ncore:]))
    else:
        from pyscf.pbc.gto.cell import intor_cross
        s2 = pcell.pbc_intor('int1e_ovlp')[baslst][:,baslst]
        s21 = intor_cross('int1e_ovlp', pcell, cell)[baslst]
        s21 = np.dot(s21, mo_coeff[:, ncore:])
    
    sa = s21.T.dot(scipy.linalg.solve(s2, s21, assume_a='pos'))

    threshold = avas_obj.threshold
    if avas_obj.openshell_option == 2:
        wocc, u = np.linalg.eigh(sa[:(nocc-ncore), :(nocc-ncore)])
        log.info('Option 2: threshold %s', threshold)
        ncas_occ = (wocc > threshold).sum()
        nelecas = (cell.nelectron - ncore * 2) - (wocc < threshold).sum() * 2
        mocore = mo_coeff[:,ncore:nocc].dot(u[:,wocc<threshold])
        mocas = mo_coeff[:,ncore:nocc].dot(u[:,wocc>=threshold])

        wvir, u = np.linalg.eigh(sa[(nocc-ncore):,(nocc-ncore):])
        ncas_vir = (wvir > threshold).sum()
        mocas = np.hstack((mocas,
                              mo_coeff[:,nocc:].dot(u[:,wvir>=threshold])))
        movir = mo_coeff[:,nocc:].dot(u[:,wvir<threshold])
        ncas = mocas.shape[1]

        occ_weights = np.hstack([wocc[wocc<threshold], wocc[wocc>=threshold]])
        vir_weights = np.hstack([wvir[wvir>=threshold], wvir[wvir<threshold]])

    elif avas_obj.openshell_option == 3:
        docc = nocc - cell.spin
        wocc, u = np.linalg.eigh(sa[:(docc-ncore),:(docc-ncore)])
        log.info('Option 3: threshold %s, num open shell %d', threshold, cell.spin)
        ncas_occ = (wocc > threshold).sum()
        nelecas = (cell.nelectron - ncore * 2) - (wocc < threshold).sum() * 2
        mocore = mo_coeff[:,ncore:docc].dot(u[:,wocc<threshold])
        mocas = mo_coeff[:,ncore:docc].dot(u[:,wocc>=threshold])

        wvir, u = np.linalg.eigh(sa[(nocc-ncore):,(nocc-ncore):])
        ncas_vir = (wvir > threshold).sum()
        mocas = np.hstack((mocas,
                              mo_coeff[:,docc:nocc],
                              mo_coeff[:,nocc:].dot(u[:,wvir>=threshold])))
        movir = mo_coeff[:,nocc:].dot(u[:,wvir<threshold])
        ncas = mocas.shape[1]

        occ_weights = np.hstack([wocc[wocc<threshold], np.ones(nocc-docc),
                                    wocc[wocc>=threshold]])
        vir_weights = np.hstack([wvir[wvir>=threshold], wvir[wvir<threshold]])
    else:
        raise RuntimeError(f'Unknown option openshell_option {avas_obj.openshell_option}')

    nalpha = (nelecas + cell.spin) // 2
    nbeta = nelecas - nalpha
    log.debug('projected occ eig %s', occ_weights)
    log.debug('projected vir eig %s', vir_weights)
    log.info('Active from occupied = %d , eig %s', ncas_occ, occ_weights[occ_weights>=threshold])
    log.info('Inactive from occupied = %d', mocore.shape[1])
    log.info('Active from unoccupied = %d , eig %s', ncas_vir, vir_weights[vir_weights>=threshold])
    log.info('Inactive from unoccupied = %d', movir.shape[1])
    log.info('Dimensions of active %d', ncas)
    log.info('# of alpha electrons %d', nalpha)
    log.info('# of beta electrons %d', nbeta)

    mofreeze = mo_coeff[:,:ncore]
    if avas_obj.canonicalize:
        def trans(c):
            if c.shape[1] == 0:
                return c
            else:
                csc = reduce(np.dot, (c.T, ovlp, mo_coeff))
                fock = np.dot(csc*mo_energy, csc.T)
                e, u = scipy.linalg.eigh(fock)
                return np.dot(c, u)
        if ncore > 0:
            mofreeze = trans(mofreeze)
        mocore = trans(mocore)
        mocas = trans(mocas)
        movir = trans(movir)
    mo = np.hstack((mofreeze, mocore, mocas, movir))
    return ncas, nelecas, mo, occ_weights, vir_weights

# For more than one-kpts
def _kernelKpoints(avas_obj):
    assert avas_obj.openshell_option != 1

    log = logger.new_logger(avas_obj)
    log.info('\n** Periodic AVAS **')

    mf = avas_obj._scf
    cell = mf.cell
    kpts = mf.kpts
    nkpts = len(kpts)
    log.info('Number of k-points: %d', nkpts)
    
    # Preallocate lists for results at each k-point
    ncas_list = []
    nelecas_list = []
    mo_list = []
    occ_weights_list = []
    vir_weights_list = []


    pcell = cell.copy()
    pcell.atom = cell._atom
    pcell.a = cell.a
    pcell.unit = 'B'
    pcell.symmetry = False
    pcell.basis = avas_obj.minao
    pcell.build(False, False)

    baslst = pcell.search_ao_label(avas_obj.aolabels)
    log.info('reference AO indices for %s %s:\n %s',
            avas_obj.minao, avas_obj.aolabels, baslst)

    # Loop over the kpoints
    for k in range(nkpts):

        if isinstance(mf, scf.uhf.UHF):
            log.note('UHF/UKS object is found.  AVAS takes alpha orbitals only')
            mo_coeff = mf.mo_coeff[k][0]
            mo_occ = mf.mo_occ[k][0]
            mo_energy = mf.mo_energy[k][0]
        else:
            mo_coeff = mf.mo_coeff[k]
            mo_occ = mf.mo_occ[k]
            mo_energy = mf.mo_energy[k]

        ncore = avas_obj.ncore
        nocc = np.count_nonzero(mo_occ != 0)
        ovlp = mf.get_ovlp()[k]

        log.info('  Total number of HF MOs  is equal to    %d' ,mo_coeff.shape[1])
        log.info('  Number of occupied HF MOs is equal to  %d', nocc)

        if avas_obj.with_iao:
            raise NotImplemented
            # from pyscf.lo import iao
            # c = iao.iao(cell, mo_coeff[:,ncore:nocc], avas_obj.minao)[:,baslst]
            # s2 = reduce(np.dot, (c.T, ovlp, c))
            # s21 = reduce(np.dot, (c.T, ovlp, mo_coeff[:, ncore:]))
        else:
            from pyscf.pbc.gto.cell import intor_cross
            s2 = mf.get_ovlp(pcell,)[k][baslst][:,baslst] # Using one defined by the mf.
            s21 = intor_cross('int1e_ovlp', pcell, cell, kpts=kpts)[k][baslst]
            s21 = np.dot(s21, mo_coeff[:, ncore:])
        
        sa = s21.conj().T.dot(scipy.linalg.solve(s2, s21, assume_a='pos'))
        
        threshold = avas_obj.threshold
        if avas_obj.openshell_option == 2:
            wocc, u = np.linalg.eigh(sa[:(nocc-ncore), :(nocc-ncore)])
            log.info('Option 2: threshold %s', threshold)
            ncas_occ = (wocc > threshold).sum()
            nelecas = (cell.nelectron - ncore * 2) - (wocc < threshold).sum() * 2
            mocore = mo_coeff[:,ncore:nocc].dot(u[:,wocc<threshold])
            mocas = mo_coeff[:,ncore:nocc].dot(u[:,wocc>=threshold])

            wvir, u = np.linalg.eigh(sa[(nocc-ncore):,(nocc-ncore):])
            ncas_vir = (wvir > threshold).sum()
            mocas = np.hstack((mocas,
                                mo_coeff[:,nocc:].dot(u[:,wvir>=threshold])))
            movir = mo_coeff[:,nocc:].dot(u[:,wvir<threshold])
            ncas = mocas.shape[1]

            occ_weights = np.hstack([wocc[wocc<threshold], wocc[wocc>=threshold]])
            vir_weights = np.hstack([wvir[wvir>=threshold], wvir[wvir<threshold]])

        elif avas_obj.openshell_option == 3:
            docc = nocc - cell.spin
            wocc, u = np.linalg.eigh(sa[:(docc-ncore),:(docc-ncore)])
            log.info('Option 3: threshold %s, num open shell %d', threshold, cell.spin)
            ncas_occ = (wocc > threshold).sum()
            nelecas = (cell.nelectron - ncore * 2) - (wocc < threshold).sum() * 2
            mocore = mo_coeff[:,ncore:docc].dot(u[:,wocc<threshold])
            mocas = mo_coeff[:,ncore:docc].dot(u[:,wocc>=threshold])

            wvir, u = np.linalg.eigh(sa[(nocc-ncore):,(nocc-ncore):])
            ncas_vir = (wvir > threshold).sum()
            mocas = np.hstack((mocas,
                                mo_coeff[:,docc:nocc],
                                mo_coeff[:,nocc:].dot(u[:,wvir>=threshold])))
            movir = mo_coeff[:,nocc:].dot(u[:,wvir<threshold])
            ncas = mocas.shape[1]

            occ_weights = np.hstack([wocc[wocc<threshold], np.ones(nocc-docc),
                                        wocc[wocc>=threshold]])
            vir_weights = np.hstack([wvir[wvir>=threshold], wvir[wvir<threshold]])
        else:
            raise RuntimeError(f'Unknown option openshell_option {avas_obj.openshell_option}')

        nalpha = (nelecas + cell.spin) // 2
        nbeta = nelecas - nalpha
        log.debug('projected occ eig %s', occ_weights)
        log.debug('projected vir eig %s', vir_weights)
        log.info('Active from occupied = %d , eig %s', ncas_occ, occ_weights[occ_weights>=threshold])
        log.info('Inactive from occupied = %d', mocore.shape[1])
        log.info('Active from unoccupied = %d , eig %s', ncas_vir, vir_weights[vir_weights>=threshold])
        log.info('Inactive from unoccupied = %d', movir.shape[1])
        log.info('Dimensions of active %d', ncas)
        log.info('# of alpha electrons %d', nalpha)
        log.info('# of beta electrons %d', nbeta)

        mofreeze = mo_coeff[:,:ncore]
        if avas_obj.canonicalize:
            def trans(c):
                if c.shape[1] == 0:
                    return c
                else:
                    csc = reduce(np.dot, (c.conj().T, ovlp, mo_coeff))
                    fock = np.dot(csc*mo_energy, csc.conj().T)
                    e, u = scipy.linalg.eigh(fock)
                    return np.dot(c, u) # No symm
            if ncore > 0:
                mofreeze = trans(mofreeze)
            mocore = trans(mocore)
            mocas = trans(mocas)
            movir = trans(movir)
        mo = np.hstack((mofreeze, mocore, mocas, movir))
        
        ncas_list.append(ncas)
        nelecas_list.append(nelecas)
        mo_list.append(mo)
        occ_weights_list.append(occ_weights)
        vir_weights_list.append(vir_weights)
    return ncas_list, nelecas_list, mo_list, occ_weights_list, vir_weights_list

@lib.with_doc(kernel.__doc__)
class AVAS(lib.StreamObject):
    def __init__(self, mf, aolabels, threshold=THRESHOLD, minao=MINAO,
                 with_iao=WITH_IAO, openshell_option=OPENSHELL_OPTION,
                 canonicalize=CANONICALIZE, ncore=0, verbose=None):
        self._scf = mf
        self.aolabels = aolabels
        self.threshold = threshold
        self.minao = minao
        self.with_iao = with_iao
        self.openshell_option = openshell_option
        self.canonicalize = canonicalize
        self.ncore = ncore
        self.stdout = mf.stdout
        self.verbose = verbose or mf.verbose
        self.ncas = None
        self.nelecas = None
        self.mo_coeff = None
        self.occ_weights = None
        self.vir_weights = None
        self.nkpts = len(mf.kpts)

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** AVAS flags ********')
        log.info('aolabels = %s', self.aolabels)
        log.info('ncore = %s', self.ncore)
        log.info('minao = %s', self.minao)
        log.info('threshold = %s', self.threshold)
        log.info('with_iao = %s', self.with_iao)
        log.info('openshell_option = %s', self.openshell_option)
        log.info('canonicalize = %s', self.canonicalize)
        log.info('nkpts = %s', self.nkpts)
        return self

    def kernel(self):
        self.dump_flags()
        totkpt = sum([np.sum(np.abs(totkpt)) for totkpt in self._scf.kpts])
        if self.nkpts == 1 and totkpt < 1e-9:
            self.ncas, self.nelecas, self.mo_coeff, \
                    self.occ_weights, self.vir_weights = _kernelGamma(self)
        elif self.nkpts >= 1 and totkpt > 1e-9:
            self.ncas, self.nelecas, self.mo_coeff, \
                    self.occ_weights, self.vir_weights = _kernelKpoints(self)
            # Some sanity checks should be here.
        else:
            raise ValueError('Number of k-points is invalid')

        return self.ncas, self.nelecas, self.mo_coeff

from pyscf.mcscf import avas as molAVAS
molAVAS = molAVAS.AVAS
pbcAVAS = AVAS