from pyscf import gto, scf, df, lib, lo, tools
import numpy as np
from scipy import linalg, sparse
import math, sys, time, ctypes
from mrh.lib.helper import load_library
from mrh.my_pyscf.df.sparse_df import sparsedf_array
import copy
libsint = load_library ('libsint')

'''
This contains a working local-density-fitting code for vj and vk which beats PySCF's default
clock speed for 15-ene with the 6-31g basis given a coarse enough population cutoff.
However, it has big errors when trying to calculate the change in vj and vk given a shifted density matrix
for some reason.
'''

def get_xyz(nmer, rCH=1.091, rCCd=1.369, rCCs=1.426, tCCC=124.5, tCCH=118.3):
    tCCC *= math.pi / 180
    tCCH *= math.pi / 180
    Hxoff = abs (rCH * math.cos (tCCH))
    Hyoff = abs (rCH * math.sin (tCCH))
    Cxoff = rCCd + abs (rCCs * math.cos (tCCC))
    Cyoff = abs (rCCs * math.sin (tCCC))
    xyz = 'H {:13.10f} {:13.10f} 0.0\n'.format (-Hxoff, -Hyoff)
    for imer in range (nmer):
        dx = imer * Cxoff
        dy = imer * Cyoff
        xyz += 'C {:13.10f} {:13.10f} 0.0\n'.format (dx, dy)
        xyz += 'H {:13.10f} {:13.10f} 0.0\n'.format (dx-Hxoff, dy+Hyoff)
        xyz += 'C {:13.10f} {:13.10f} 0.0\n'.format (dx+rCCd, dy)
        xyz += 'H {:13.10f} {:13.10f} 0.0\n'.format (dx+rCCd+Hxoff, dy-Hyoff)
    xyz += 'H {:13.10f} {:13.10f} 0.0'.format (dx+rCCd+Hxoff, dy+Hyoff)
    return xyz

def schmidt (dm, ovlp, mo_occ, idx_frag):
    smo = ovlp[:,idx_frag] @ mo_occ[idx_frag]
    smoH = smo.conjugate ().T
    wgt, umat = linalg.eigh (smoH @ dm @ smo)
    mo_occ = mo_occ @ umat[:,np.abs (wgt) > 1e-8]
    return mo_occ

def get_pops (mol, ovlp, mo_coeff):
    ao_offset = mol.offset_ao_by_atom ()
    pops = np.zeros (mol.natm)
    for iatm in range (mol.natm):
        i = ao_offset[iatm,2]
        j = ao_offset[iatm,3]
        ovlp_iatm = ovlp[i:j,i:j]
        mo_iatm = mo_coeff[i:j,:]
        pops[iatm] = ((ovlp_iatm @ mo_iatm) * mo_iatm).sum ()
    npops = linalg.norm (pops)
    if npops > 0: pops /= npops
    return pops

def imp_cholesky (mf, mol, auxmol, j3c, j2c, pops):
    t0 = (time.clock (), time.time ())
    nao = mol.nao_nr ()
    naux = auxmol.nao_nr ()
    idx_impao = np.zeros (nao, dtype=np.bool_)
    idx_impaux = np.zeros (naux, dtype=np.bool_)
    idx_atms = np.where (pops > 0.001)[0]
    ao_offs = mol.offset_ao_by_atom ()
    aux_offs = auxmol.offset_ao_by_atom ()
    for iatm in idx_atms:
        idx_impao[ao_offs[iatm,2]:ao_offs[iatm,3]] = True
        idx_impaux[aux_offs[iatm,2]:aux_offs[iatm,3]] = True
    idx_impao_2d = np.zeros ((nao, nao), dtype=idx_impao.dtype)
    idx_impao_2d[np.ix_(idx_impao, idx_impao)] = True
    idx_impao_2dtril = idx_impao_2d[np.tril_indices (nao)]
    t1 = lib.logger.timer (mf, 'making index arrays', *t0)
    j2c_imp = j2c[np.ix_(idx_impaux,idx_impaux)]
    t1 = lib.logger.timer (mf, 'copying j2c', *t1)
    j3c_imp = j3c[np.ix_(idx_impao_2dtril,idx_impaux)]
    t1 = lib.logger.timer (mf, 'copying j3c', *t1)
    try:
        low = linalg.cholesky(j2c_imp, lower=True)
        j2c_imp = None
        cderi = linalg.solve_triangular(low, j3c_imp.T, lower=True,
                                              overwrite_b=True)
    except linalg.LinAlgError:
        w, v = linalg.eigh(j2c_imp)
        idx = w > LINEAR_DEP_THR
        v = (v[:,idx] / numpy.sqrt(w[idx]))
        cderi = lib.dot(v.T, j3c_imp.T)
    t1 = lib.logger.timer (mf, 'the actual cholesky linear algebra', *t1)
    if cderi.flags.f_contiguous:
        cderi = lib.transpose(cderi.T)
    t1 = lib.logger.timer (mf, 'cholesky transposing', *t1)
    return cderi, idx_impao, idx_impaux, idx_impao_2dtril


def sparse_jk (mf, mol, cderi, atom_clusters, dm, auxmol, j3c, j2c):
    t0 = (time.clock (), time.time ())
    ao_offset = mol.offset_ao_by_atom ()
    ovlp = mol.intor ('int1e_ovlp')
    mo_occ, mo_coeff = linalg.eigh (-ovlp @ dm @ ovlp, b = ovlp)
    mo_occ = -mo_occ
    naux = cderi.shape[0]
    nao = dm.shape[-1]
    ncore = mol.nelectron // 2
    rho = np.zeros (naux, dtype=dm.dtype)
    vk = np.zeros ((nao, nao), dtype=dm.dtype)
    t1 = lib.logger.timer (mf, 'preparation', *t0)
    dm_rho = np.zeros (nao*(nao+1)//2, dtype=dm.dtype)
    for frag in atom_clusters:
        idx_frag = np.zeros (nao, dtype=np.bool_)
        for iatm in frag:
            idx_frag[ao_offset[iatm,2]:ao_offset[iatm,3]] = True
        imp_coeff = schmidt (dm, ovlp, mo_coeff[:,:ncore], idx_frag)
        smo = ovlp @ imp_coeff
        smoH = smo.conjugate ().T
        imp_occ, umat = linalg.eigh (smoH @ dm @ smo)
        umat = umat[:,np.abs(imp_occ)>1e-8]
        imp_occ = imp_occ[np.abs(imp_occ)>1e-8]
        imp_coeff = imp_coeff @ umat
        frag_coeff = imp_coeff.copy ()
        frag_coeff[~idx_frag,:] = 0.0
        t1 = lib.logger.timer (mf, 'schmidt', *t1)
        pops = get_pops (mol, ovlp, imp_coeff * np.abs (imp_occ[None,:]))
        t1 = lib.logger.timer (mf, 'pops', *t1)
        cderi_impao, idx_impao, idx_impaux, idx_rho = imp_cholesky (mf, mol, auxmol, j3c, j2c, pops)
        t1 = lib.logger.timer (mf, 'imp cholesky', *t1)
        impao_coeff = imp_coeff[idx_impao,:]
        fragao_coeff = frag_coeff[idx_impao,:]
        dm_imp = np.dot (imp_coeff * imp_occ[None,:], frag_coeff.conjugate ().T)
        dm_imp += dm_imp.T
        dm_imp /= 2
        dm_imp_tril = lib.pack_tril (dm_imp + dm_imp.T - np.diag (np.diag (dm_imp)))
        dm_rho += dm_imp_tril
        t1 = lib.logger.timer (mf, 'rho', *t1)

        dm_rect = np.dot (imp_coeff[idx_frag,:] * imp_occ[None,:], imp_coeff.conjugate ().T).T[idx_impao,:]
        cderi_impao = sparsedf_array (cderi_impao)
        idx_vk = np.ix_(idx_impao,idx_impao)
        vPuv_impao = cderi_impao.contract1 (dm_rect)
        t1 = lib.logger.timer (mf, 'vk first dot', *t1)
        bPuv_impao = cderi_impao.unpack_mo ()[:,idx_frag[idx_impao],:]
        t1 = lib.logger.timer (mf, 'vk indexing', *t1)
        #print (vPuv_impao.shape)
        #vPuv_impao = np.ascontiguousarray (vPuv_impao.transpose (1,0,2))
        #t1 = lib.logger.timer (mf, 'vk transpose', *t1)
        vk[idx_vk] += np.tensordot (bPuv_impao, vPuv_impao, axes=((1,0),(1,2)))
        t1 = lib.logger.timer (mf, 'vk second dot', *t1)
    rho = np.dot (cderi, dm_rho)
    vj = lib.unpack_tril (np.dot (rho, cderi))
    t1 = lib.logger.timer (mf, 'vj', *t1)
    return vj, vk

if len (sys.argv) > 2: thresh = float (sys.argv[2])
n = int (sys.argv[1])
calcname = 'polyene_{}mer_dfhf_631g'.format (n)
lib.logger.TIMER_LEVEL = lib.logger.INFO
mol = gto.M (atom = get_xyz (n), basis = '6-31g', symmetry=False, verbose = lib.logger.INFO, output = calcname + '.log')
my_aux = df.aug_etb (mol)
mf = scf.RHF (mol).density_fit (auxbasis = my_aux)
mf.max_cycle = 1
mf.kernel ()

atom_clusters = [list (range (1))]
offs = 1
step = 1
for imer in range (1, mol.natm):
    atom_clusters.append (list (range (offs, offs+step)))
    offs += step
#atom_clusters.append (range (offs, offs+5))

nao = mol.nao_nr ()
dm = mf.make_rdm1 ()
ovlp = mf.get_ovlp ()
evals, evecs = linalg.eigh (ovlp @ dm @ ovlp, b=ovlp)
evals[np.abs (evals) < 1e-8] = 0.0
#dm = lib.tag_array (dm, mo_coeff=evecs, mo_occ=evals)
mo_coeff = mf.mo_coeff
mo_occ = mf.mo_occ
t0, w0 = time.clock (), time.time()
j3c = df.incore.aux_e2 (mol, mf.with_df.auxmol, aosym='s2ij')
j2c = df.incore.fill_2c2e (mol, mf.with_df.auxmol)
t1, w1 = time.clock (), time.time()
print (j3c.shape, j2c.shape, t1 - t0, w1 - w0)
vj_ref, vk_ref = mf.with_df.get_jk (dm=dm)

t0 = (time.clock (), time.time ())
vj, vk = sparse_jk (mf, mol, mf.with_df._cderi.copy (), copy.deepcopy (atom_clusters), dm.copy (), mf.with_df.auxmol, j3c.copy (), j2c.copy ())
t1 = lib.logger.timer (mf, 'sparse df vj and vk', *t0)
ix_iao, ix_jao = np.tril_indices (nao)
err_j = linalg.norm (vj[(ix_iao,ix_jao)] - vj_ref[(ix_iao,ix_jao)]) / (nao * (nao+1) // 2)
err_k = linalg.norm (vk[(ix_iao,ix_jao)] - vk_ref[(ix_iao,ix_jao)]) / (nao * (nao+1) // 2)
avg_j = linalg.norm (vj[(ix_iao,ix_jao)]) / (nao * (nao+1) // 2)
avg_k = linalg.norm (vk[(ix_iao,ix_jao)]) / (nao * (nao+1) // 2)
lib.logger.info (mf, "error_J = {}; avg_J = {}".format (err_j, avg_j))
lib.logger.info (mf, "error_K = {}; avg_K = {}".format (err_k, avg_k))

fock = mf.get_hcore () + vj - vk/2
mo_ene, mo_coeff = linalg.eigh (fock, b=mf.get_ovlp ())
ncore = mol.nelectron // 2
idx = np.argsort (mo_ene)
mo_ene = mo_ene[idx]
mo_coeff = mo_coeff[:,idx]
dm_new = 2 * mo_coeff[:,:ncore] @ mo_coeff[:,:ncore].T
ddm = dm_new - dm
vj_ref, vk_ref = mf.with_df.get_jk (dm=ddm)
vj, vk = sparse_jk (mf, mol, mf.with_df._cderi, atom_clusters, ddm, mf.with_df.auxmol, j3c, j2c)
err_j = linalg.norm (vj[(ix_iao,ix_jao)] - vj_ref[(ix_iao,ix_jao)]) / (nao * (nao+1) // 2)
err_k = linalg.norm (vk[(ix_iao,ix_jao)] - vk_ref[(ix_iao,ix_jao)]) / (nao * (nao+1) // 2)
avg_j = linalg.norm (vj[(ix_iao,ix_jao)]) / (nao * (nao+1) // 2)
avg_k = linalg.norm (vk[(ix_iao,ix_jao)]) / (nao * (nao+1) // 2)
lib.logger.info (mf, "error_J = {}; avg_J = {}".format (err_j, avg_j))
lib.logger.info (mf, "error_K = {}; avg_K = {}".format (err_k, avg_k))


'''
npair_tot = nao * (nao + 1) // 2
nocc = mol.nelectron // 2
data = [nao, npair_tot]
naux = mf.with_df._cderi.shape[0]
cderi = mf.with_df._cderi.copy ()
metric = linalg.norm (cderi, axis=0)
cderi[:,metric < 1e-8] = 0.0
eri = sparse.csr_matrix (cderi)
eri.eliminate_zeros ()
dm_tril = lib.pack_tril (dm + dm.T - np.diag (np.diag (dm)))
rho = eri * dm_tril #np.dot (cderi, dm_tril)
vj = lib.unpack_tril (rho * eri)
eri = sparse.csr_matrix (lib.unpack_tril (cderi).reshape (-1, nao))
vPuv = (eri * dm).reshape (naux, nao, nao).transpose (0,2,1).reshape (naux*nao, nao)
eri = sparse.csc_matrix (lib.unpack_tril (cderi).transpose (1,0,2).reshape (nao, naux*nao))
vk = eri * vPuv
#wrk1 = cderi.contract1 (dm)
#wrk1 = np.ascontiguousarray (wrk1.transpose (1,0,2))
#vk = cderi.contract2 (wrk1)
t1 = lib.logger.timer (mf, 'sparse df vj and vk', *t0)
err_j = linalg.norm (vj[(ix_iao,ix_jao)] - vj_ref[(ix_iao,ix_jao)]) / (nao * (nao+1) // 2)
err_k = linalg.norm (vk[(ix_iao,ix_jao)] - vk_ref[(ix_iao,ix_jao)]) / (nao * (nao+1) // 2)
lib.logger.info (mf, "error_J = {}".format (err_j))
lib.logger.info (mf, "error_K = {}".format (err_k))


blo = lo.Boys (mol, mf.mo_coeff[:,:nocc]).kernel ()
idx = np.power (blo, 2) < 1e-8
blo[idx] = 0
ovlp = blo.conjugate ().T @ mf.get_ovlp () @ blo
blo = np.dot (blo, lo.orth.lowdin (ovlp))
cderi = cderi.unpack_mo ()
cderi = np.dot (cderi, blo)
metric = linalg.norm (cderi, axis=0)
print (np.amax (np.count_nonzero (metric > 1e-8, axis=0)))
print (np.amax (np.count_nonzero (metric > 1e-8, axis=1)))
print (metric.size, np.count_nonzero (metric > 1e-8))

#test_wrk1 = lib.unpack_tril (mf.with_df._cderi).transpose (1,2,0)
#test_wrk1 = np.dot (dm, test_wrk1)
#print ("First contraction:")
#print ("dm, cderi, aux: {}".format (linalg.norm (wrk1.reshape (-1) - test_wrk1.ravel ())))
#print ("cderi, dm, aux (what it's SUPPOSED to be): {}".format (linalg.norm (wrk1.reshape (-1) - test_wrk1.transpose (1,0,2).ravel ())))
#print ("dm, aux, cderi: {}".format (linalg.norm (wrk1.reshape (-1) - test_wrk1.transpose (0,2,1).ravel ())))
#print ("cderi, aux, dm: {}".format (linalg.norm (wrk1.reshape (-1) - test_wrk1.transpose (1,2,0).ravel ())))
#print ("aux, dm, cderi: {}".format (linalg.norm (wrk1.reshape (-1) - test_wrk1.transpose (2,0,1).ravel ())))
#print ("aux, cderi, dm: {}".format (linalg.norm (wrk1.reshape (-1) - test_wrk1.transpose (2,1,0).ravel ())))
#t1 = lib.logger.timer (mf, 'sparse vk first comparison', *t1)
'''


# May not be the right approach because stuff on the same atom has different principle quantum numbers
#''' overlap of normalized 1s with primitives zm, zn and distance R should be:
#    zt^(3/2) (zn zm)^(-3/4) exp [-zt R^2]
#    where zt = (zn zm) / (zn + zm)
#    strange second factor because 1s is normalized? IDK I need to make sure this is right. '''
#thresh = 1e-8
#t0 = (time.clock (), time.time ())
#p = np.zeros ((nao, nao), dtype=np.bool_)
#iao = jao = 0
#for ishell, idata in enumerate (mol._bas):
#    iatm, il = idata[:2]
#    iprim = mol.bas_exp (ishell)
#    icoord = mol.bas_coord (ishell)
#    ix_i = list (range (iao, iao+((2*il)+1)))
#    for jshell, jdata in enumerate (mol._bas[:ishell+1]):
#        jatm, jl = jdata[:2]
#        jprim = mol.bas_exp (jshell)
#        jcoord = mol.bas_coord (jshell)
#        ix_j = list (range (jao, jao+((2*jl)+1)))
#        ###
#        R = linalg.norm (icoord - jcoord)
#        zprod = np.multiply.outer (iprim, jprim)
#        zrat = zprod / np.add.outer (iprim, jprim)
#        ovlp = np.exp (-zrat * (R**2))
#        ovlp *= np.power (zrat, 1.5)
#        ovlp *= np.power (zprod, -0.75)
#        ###
#        p[np.ix_(ix_i,ix_j)] = np.count_nonzero (np.abs (ovlp) > 1e-13)
#        jao += (2*jl)+1
#    iao += (2*il)+1
#    jao = 0
#metric = p[np.tril_indices (nao)]
#npair_sparse = np.count_nonzero (metric)
#idx_nonzero = np.where (metric)[0]
#metric = linalg.norm (mf.with_df._cderi, axis=0)
#idx_nonzero = metric > thresh
#npair_sparse = np.count_nonzero (idx_nonzero)
#idx_nonzero = np.where (idx_nonzero)[0]
#t1 = lib.logger.timer (mf, 'sparsity index', *t0)
#metric = lib.unpack_tril (metric > thresh).astype (np.int32)
#iao_nent = np.count_nonzero (metric, axis=0).astype (np.int32)
#iao_sort = np.argsort (iao_nent).astype (np.int32)
#nent_max = np.amax (iao_nent)
#iao_entlist = -np.ones ((nao, nent_max), dtype=np.int32)
#for iao, nent in enumerate (iao_nent):
#    iao_entlist[iao,:nent] = np.where (metric[iao,:])[0]
#
#vj_ref, vk_ref = mf.with_df.get_jk (dm=dm)
#vj_ref = vj_ref[np.tril_indices (nao)]
#t0 = (time.clock (), time.time ())
#naux, npair_dense = mf.with_df._cderi.shape
#cderi_sparse = mf.with_df._cderi[:,idx_nonzero]
#t1 = lib.logger.timer (mf, 'sparsify cderi array', *t0)
## J
#dm_sparse = dm + dm.T
#dm_sparse[np.diag_indices (nao)] /= 2
#dm_sparse = dm_sparse[np.tril_indices (nao)][idx_nonzero]
#rho = np.dot (cderi_sparse, dm_sparse)
#tvj = np.dot (rho, cderi_sparse)
#vj = np.zeros (nao*(nao+1)//2)
#vj[idx_nonzero] = tvj
#t1 = lib.logger.timer (mf, 'sparse vj crunching', *t1)
## K
#vk = np.zeros ((nao, nao), dtype=dm.dtype)
#wrk1 = np.zeros (nao*nao*naux, dtype=dm.dtype)
#wrk2 = np.zeros (lib.num_threads () * naux, dtype=dm.dtype)
#wrk3 = np.zeros (lib.num_threads () * nent_max * naux, dtype = dm.dtype)
#wrk4 = np.zeros (lib.num_threads () * nent_max * nao, dtype = dm.dtype)
#idx_nonzero = idx_nonzero.astype (np.int32)
#ix_iao = ix_iao.astype (np.int32)
#ix_jao = ix_jao.astype (np.int32)
#t1 = lib.logger.timer (mf, 'sparse vk throat-clearing', *t1)
#libsint.SINT_SDCDERI_DDMAT (mf.with_df._cderi.ctypes.data_as (ctypes.c_void_p),
#    dm.ctypes.data_as (ctypes.c_void_p),
#    wrk1.ctypes.data_as (ctypes.c_void_p),
#    wrk3.ctypes.data_as (ctypes.c_void_p),
#    wrk4.ctypes.data_as (ctypes.c_void_p),
#    iao_sort.ctypes.data_as (ctypes.c_void_p),
#    iao_nent.ctypes.data_as (ctypes.c_void_p),
#    iao_entlist.ctypes.data_as (ctypes.c_void_p),
#    ctypes.c_int (nao),
#    ctypes.c_int (naux),
#    ctypes.c_int (nao),
#    ctypes.c_int (nent_max))
#libsint.SDFKmatR1 (cderi_sparse.ctypes.data_as (ctypes.c_void_p),
#    dm.ctypes.data_as (ctypes.c_void_p),
#    wrk1.ctypes.data_as (ctypes.c_void_p),
#    idx_nonzero.ctypes.data_as (ctypes.c_void_p),
#    ix_iao.ctypes.data_as (ctypes.c_void_p),
#    ix_jao.ctypes.data_as (ctypes.c_void_p),
#    ctypes.c_int (npair_sparse),
#    ctypes.c_int (nao),
#    ctypes.c_int (naux)) 
#t2 = lib.logger.timer (mf, 'sparse vk first dot product', *t1)
#t2 = lib.logger.timer (mf, 'comparison', *t2)
#libsint.SDFKmatRT (wrk1.ctypes.data_as (ctypes.c_void_p),
#    wrk2.ctypes.data_as (ctypes.c_void_p),
#    ix_iao.ctypes.data_as (ctypes.c_void_p),
#    ix_jao.ctypes.data_as (ctypes.c_void_p),
#    ctypes.c_int (nao),
#    ctypes.c_int (naux)) 
#t2 = lib.logger.timer (mf, 'sparse vk transpose', *t2)
#print ("Transpose:")
#print ("dm, cderi, aux (what it's SUPPOSED to be): {}".format (linalg.norm (wrk1 - test_wrk1.ravel ())))
#print ("cderi, dm, aux: {}".format (linalg.norm (wrk1 - test_wrk1.transpose (1,0,2).ravel ())))
#print ("dm, aux, cderi: {}".format (linalg.norm (wrk1 - test_wrk1.transpose (0,2,1).ravel ())))
#print ("cderi, aux, dm: {}".format (linalg.norm (wrk1 - test_wrk1.transpose (1,2,0).ravel ())))
#print ("aux, dm, cderi: {}".format (linalg.norm (wrk1 - test_wrk1.transpose (2,0,1).ravel ())))
#print ("aux, cderi, dm: {}".format (linalg.norm (wrk1 - test_wrk1.transpose (2,1,0).ravel ())))
#t2 = lib.logger.timer (mf, 'comparison', *t2)
#libsint.SDFKmatR2 (cderi_sparse.ctypes.data_as (ctypes.c_void_p),
#    wrk1.ctypes.data_as (ctypes.c_void_p),
#    vk.ctypes.data_as (ctypes.c_void_p),
#    idx_nonzero.ctypes.data_as (ctypes.c_void_p),
#    ix_iao.ctypes.data_as (ctypes.c_void_p),
#    ix_jao.ctypes.data_as (ctypes.c_void_p),
#    ctypes.c_int (npair_sparse),
#    ctypes.c_int (nao),
#    ctypes.c_int (naux))
#t2 = lib.logger.timer (mf, 'sparse vk final operation', *t2)
#wrk1[:] = wrk2[:] = vk[:] = 0.0
#libsint.SDFKmatR (cderi_sparse.ctypes.data_as (ctypes.c_void_p),
#    dm.ctypes.data_as (ctypes.c_void_p),
#    vk.ctypes.data_as (ctypes.c_void_p),
#    wrk1.ctypes.data_as (ctypes.c_void_p),
#    wrk2.ctypes.data_as (ctypes.c_void_p),
#    idx_nonzero.ctypes.data_as (ctypes.c_void_p),
#    ix_iao.ctypes.data_as (ctypes.c_void_p),
#    ix_jao.ctypes.data_as (ctypes.c_void_p),
#    ctypes.c_int (npair_sparse),
#    ctypes.c_int (nao),
#    ctypes.c_int (naux)) 
#t1 = lib.logger.timer (mf, 'sparse vk crunching (total)', *t1)
#err_j = linalg.norm (vj - vj_ref) / (nao * (nao+1) // 2)
