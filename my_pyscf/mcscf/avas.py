from pyscf.mcscf.avas import *

def expand_mc (mc, aolabels, **kwargs):
    mf = mc._scf.copy ()
    mf.mo_coeff = mc.mo_coeff.copy ()
    mf.mo_occ = numpy.zeros (mf.mo_coeff.shape[1], dtype=int)
    mf.mo_occ[:(mc.ncore+mc.ncas)] += 1
    mf.mo_occ[:mc.ncore] += 1
    fname = kwargs.pop ('molden', None)
    kwargs['openshell_option'] = 3
    with lib.temporary_env (mf.mol, spin=mc.ncas):
        ncas, nelecas, mo_coeff = kernel (mf, aolabels, **kwargs)
    if fname:
        molden (mc, fname, ncas, nelecas, mo_coeff)
    return ncas, nelecas, mo_coeff

def molden (mc, fname, ncas, nelecas, mo_coeff):
    from pyscf.tools.molden import from_mo
    from pyscf.mcscf import CASCI
    mc1 = CASCI (mc._scf, ncas, nelecas)
    i = mc1.ncore
    j = mc.ncore
    k = j + mc.ncas
    l = i + mc1.ncas
    assert (i<=j), '{} {} {} {}'.format (i,j,k,l)
    assert (k<=l), '{} {} {} {}'.format (i,j,k,l)
    mo_occ = numpy.zeros (mo_coeff.shape[1], dtype=int)
    mo_occ[:j] = 2
    mo_occ[j:k] = 1
    mo_coeff = mo_coeff[:,i:l]
    mo_occ = mo_occ[i:l]
    from_mo (mc.mol, fname, mo_coeff, occ=mo_occ)
    return 


