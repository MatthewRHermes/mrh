import h5py
from pyscf.lib.chkfile import load
from pyscf.lib.chkfile import load_mol, save_mol

def load_las_(mc, chkfile=None):
    if chkfile is None: chkfile = mc.chkfile
    if chkfile is None: raise RuntimeError ('chkfile not specified')
    data = load (chkfile, 'las')
    if data is None: raise KeyError ('LAS record not in chkfile')
    mc.mo_coeff = data['mo_coeff']
    ci = data['ci']
    mc.ci = []
    for i in range (mc.nfrags):
        mc.ci.append ([])
        cii = ci[str(i)]
        for j in range (mc.nroots):
            mc.ci[-1].append (cii[str(j)])
    return mc

def dump_las (mc, chkfile=None, key='las', mo_coeff=None, ci=None,
              overwrite_mol=True):
    if chkfile is None: chkfile = mc.chkfile
    if not chkfile: return mc
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci

    with h5py.File (chkfile, 'a') as fh5:
        if 'mol' not in fh5:
            fh5['mol'] = mc.mol.dumps()
        elif overwrite_mol:
            del (fh5['mol'])
            fh5['mol'] = mc.mol.dumps()
        if key in fh5:
            del (fh5[key])
        data = fh5.create_group (key)

        data['mo_coeff'] = mo_coeff
        for i, cii in enumerate (ci):
            data_ci_i = data.create_group ('ci/'+str(i))
            for j, ciij in enumerate (cii):
                data_ci_i[str(j)] = ciij
    return mc


if __name__=='__main__':
    from pyscf import gto, scf, mcscf
    from mrh.my_pyscf.mcscf.lasscf_sync_o0 import LASSCF
    xyz='''Li 0 0 0,
           H 2 0 0,
           Li 10 0 0,
           H 12 0 0'''
    mol = gto.M (atom=xyz, basis='6-31g', symmetry=False, output='chkfile.log')
    mf = scf.RHF (mol).run ()
    las = LASSCF (mf, (2,2), (2,2))
    mc = mcscf.CASSCF (mf, 4, 4).run ()
    mo = las.localize_init_guess (([0,1],[2,3]), mc.mo_coeff, freeze_cas_spaces=True)
    las.kernel (mo)
    dump_las (las, chkfile='chkfile.chk')
    load_las_(las, 'chkfile.chk')



