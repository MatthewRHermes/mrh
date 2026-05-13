import numpy as np
import scipy
from functools import reduce

from pyscf import lib, __config__
from pyscf.lib import logger
from pyscf.pbc import gto, scf
from pyscf.mcscf import avas as molAVAS
from pyscf.pbc.gto.cell import intor_cross

'''
Atomic valence active orbitals (AVAS)
Ref. J. Chem. Theory Comput. 2017, 13, 4063−4078

Here, I have adapting the AVAS algorithm for the selection of active space 
with the PBC MCSCF (k-point sampling as well). 

I can not use the molecular AVAS off the shelf because the AVAS uses AOs which are defined 
for the molecule and not for the periodic system, second the overlap is also computed for 
molecule not for the periodic system. Third and most important it can not be used with
k-point sampling (as in kmf) and complex numbers. So, I have to adapt the AVAS algorithm 
for the periodic system.
'''

# Just call from molecular AVAS settings.
THRESHOLD = molAVAS.THRESHOLD
MINAO = molAVAS.MINAO
WITH_IAO = molAVAS.WITH_IAO
OPENSHELL_OPTION = molAVAS.OPENSHELL_OPTION
CANONICALIZE = molAVAS.CANONICALIZE

@lib.with_doc(molAVAS.kernel.__doc__)
def kernel(mf, aolabels, threshold=THRESHOLD, minao=MINAO, with_iao=WITH_IAO,
           openshell_option=OPENSHELL_OPTION, canonicalize=CANONICALIZE,
           ncore=0, verbose=None):
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

def _kernel(avas_obj, mf, mo_coeff, mo_occ, mo_energy, ovlp, log, baslst, pcell, cell, kpts=None, k=None):
    '''
    General kernel for the p-AVAS.
    '''
    
    if isinstance(mf, scf.kuhf.KUHF):
        log.note('UHF/UKS object is found.  AVAS takes alpha orbitals only')
        mo_coeff = np.asarray(mo_coeff[0])
        mo_occ = np.asarray(mo_occ[0])
        mo_energy = np.asarray(mo_energy[0])
        print('mo_coeff shape', mo_coeff.shape)
    else:
        mo_coeff = np.asarray(mo_coeff)
        mo_occ = np.asarray(mo_occ)
        mo_energy = np.asarray(mo_energy)

    ncore = avas_obj.ncore
    nocc = np.count_nonzero(mo_occ != 0)

    if mo_coeff.ndim > 2:
        assert mo_coeff.ndim == 3 and mo_coeff.shape[0] == 1, \
        'The shape of mo_coeff is not expected'
        mo_coeff = mo_coeff[0]
    if mo_occ.ndim > 1:
        assert mo_occ.ndim == 2 and mo_occ.shape[0] == 1, \
        'The shape of mo_occ is not expected'
        mo_occ = mo_occ[0]
    if mo_energy.ndim > 1:
        assert mo_energy.ndim == 2 and mo_energy.shape[0] == 1, \
        'The shape of mo_energy is not expected'
        mo_energy = mo_energy[0]

    log.info('  Total number of HF MOs  is equal to    %d' ,mo_coeff.shape[1])
    log.info('  Number of occupied HF MOs is equal to  %d', nocc)

    if avas_obj.with_iao:
        raise NotImplementedError('IAO is not implemented for p-AVAS yet')
    else:
        s2 = ovlp[baslst][:,baslst]
        if kpts is not None:
            s21 = intor_cross('int1e_ovlp', pcell, cell, kpts=kpts)[k][baslst]
        else:
            s21 = intor_cross('int1e_ovlp', pcell, cell)[baslst]
        
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
                fock = np.dot(csc, np.dot(np.diag(mo_energy.astype(c.dtype)), csc.conj().T))
                e, u = scipy.linalg.eigh(fock)
                return np.dot(c, u) # No symm
        if ncore > 0:
            mofreeze = trans(mofreeze)
        mocore = trans(mocore)
        mocas = trans(mocas)
        movir = trans(movir)
    mo = np.hstack((mofreeze, mocore, mocas, movir))
    return ncas, nelecas, mo, occ_weights, vir_weights

def _kernelGamma(avas_obj):
    '''
    The Gamma point kmf is stored as the molecular mf object, that's why I can't use the general 
    k-point AVAS for that. I have to write the separate function for that.
    '''

    log = logger.new_logger(avas_obj)
    log.info('\n** AVAS **')
    
    mf = avas_obj._scf
    cell = mf.cell
    nkpts = len(mf.kpts)

    assert nkpts == 1, 'For more than one kpts, use the other _kernel function'

    assert avas_obj.openshell_option != 1

    # Note that after the cell construction atom coordinates and the lattice vectors are
    # stored in the 'Bohr' unit. That's why I used bohr here.
    pcell = cell.copy()
    pcell.atom = cell._atom
    pcell.a = cell.lattice_vectors()
    pcell.unit = 'B'
    pcell.pseudo = cell.pseudo
    pcell.ecp = cell.ecp
    pcell.symmetry = False
    pcell.basis = avas_obj.minao
    pcell.build(False, False)

    ovlp = mf.get_ovlp()
    if ovlp.ndim == 3: # In case of shape mismatch
        ovlp = ovlp[0]
    baslst = pcell.search_ao_label(avas_obj.aolabels)
    log.info('reference AO indices for %s %s:\n %s',
             avas_obj.minao, avas_obj.aolabels, baslst)
    return _kernel(avas_obj, mf, mf.mo_coeff, mf.mo_occ, mf.mo_energy, ovlp, log, baslst, pcell, cell)

def _kernelKpoints(avas_obj):
    '''
    Kernel function for k-point AVAS
    '''
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

    # Note that after the cell construction atom coordinates and the lattice vectors are
    # stored in the 'Bohr' unit. That's why I used bohr here.
    pcell = cell.copy()
    pcell.atom = cell._atom
    pcell.a = cell.lattice_vectors()
    pcell.unit = 'B'
    pcell.pseudo = cell.pseudo
    pcell.ecp = cell.ecp
    pcell.symmetry = False
    pcell.basis = avas_obj.minao
    pcell.build(False, False)

    baslst = pcell.search_ao_label(avas_obj.aolabels)
    log.info('reference AO indices for %s %s:\n %s',
            avas_obj.minao, avas_obj.aolabels, baslst)

    def _get_ovlp_k(k):
        return mf.get_ovlp(cell, kpts[k])
    
    for k in range(nkpts):
        if isinstance(mf, scf.kuhf.KUHF):
            ncas, nelecas, mo, occ_weights, vir_weights = \
                _kernel(avas_obj, mf, mf.mo_coeff[:, k], mf.mo_occ[:, k], 
                        mf.mo_energy[:, k], _get_ovlp_k(k), log, baslst, pcell, cell, kpts=kpts, k=k)
        else:
            ncas, nelecas, mo, occ_weights, vir_weights = \
                _kernel(avas_obj, mf, mf.mo_coeff[k], mf.mo_occ[k],
                        mf.mo_energy[k], _get_ovlp_k(k), log, baslst, pcell, cell, kpts=kpts, k=k)
        ncas_list.append(ncas)
        nelecas_list.append(nelecas)
        mo_list.append(mo)
        occ_weights_list.append(occ_weights)
        vir_weights_list.append(vir_weights)
    
    # The mo_list should be returned as np array, as that would be easier to handle in kCASSCF code.
    mo_list = np.array(mo_list)
    return ncas_list, nelecas_list, mo_list, occ_weights_list, vir_weights_list

@lib.with_doc(kernel.__doc__)
class AVAS(molAVAS.AVAS):
    def __init__(self, mf, aolabels, threshold=THRESHOLD, minao=MINAO,
                 with_iao=WITH_IAO, openshell_option=OPENSHELL_OPTION,
                 canonicalize=CANONICALIZE, ncore=0, verbose=None):
        super().__init__(mf, aolabels, threshold, minao, with_iao, 
                         openshell_option, canonicalize, ncore, verbose)
        self.nkpts = len(mf.kpts)

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        super().dump_flags(verbose)
        log.info('nkpts = %s', self.nkpts)
        return self

    def kernel(self):
        '''
        The kernel function for p-AVAS. It detects whether the underline object is cell or mol,
        and then calls the appropriate kernel function.
        Note: in case of gamma point, the return values are
        ncas: int
        nelecas: tuple of (nalpha, nbeta)
        mo_coeff: ndarray of shape (nao, nmo)
        
        In case of k-points, the return values are lists
        ncas_list: list of int (nkpts,)
        nelecas_list: list of nkpts elements, each one is tuple of (nalpha, nbeta)
        mo_coeff_list: list of nkpts elements, each one is ndarray of shape (nao, nmo)
        '''
        self.dump_flags()
        totkpt = sum([np.sum(np.abs(totkpt)) for totkpt in self._scf.kpts])
        if self.nkpts == 1 and totkpt < 1e-9:
            self.ncas, self.nelecas, self.mo_coeff, \
                    self.occ_weights, self.vir_weights = _kernelGamma(self)
        elif self.nkpts >= 1 and totkpt > 1e-9:
            self.ncas, self.nelecas, self.mo_coeff, \
                    self.occ_weights, self.vir_weights = _kernelKpoints(self)
        else:
            raise ValueError('Number of k-points is invalid')
        return self.ncas, self.nelecas, self.mo_coeff

molAVAS = molAVAS.AVAS
pbcAVAS = AVAS
