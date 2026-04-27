import h5py
from pyscf.lib.chkfile import load
from mrh.my_pyscf.mcscf import chkfile as las_chkfile
from mrh.my_pyscf.lassi import chkfile as lsi_chkfile

KEYS_CONFIG_LASSIS = lsi_chkfile.KEYS_CONFIG_LASSI
KEYS_SACONSTR_LASSIS = lsi_chkfile.KEYS_SACONSTR_LASSI
KEYS_RESULTS_LASSIS = lsi_chkfile.KEYS_RESULTS_LASSI + ['converged', 'max_disc_sval']

def load_lsis_(lsis, chkfile=None, method_key='lsi',
               keys_config=KEYS_CONFIG_LASSIS,
               keys_saconstr=KEYS_SACONSTR_LASSIS,
               keys_results=KEYS_RESULTS_LASSIS):
    lsis._las = las_chkfile.load_las_(lsis._las, chkfile=chkfile, method_key='las')

    if chkfile is None: chkfile = lsis.chkfile
    if chkfile is None: raise RuntimeError ('chkfile not specified')
    data = load (chkfile, method_key)
    if data is None: raise KeyError ('{} record not in chkfile'.format (method_key.upper()))

    lsis = las_chkfile._load_las_1_(lsis, data,
                                    keys_config=keys_config,
                                    keys_saconstr=keys_saconstr,
                                    keys_results=keys_results)
    lsis = _load_lsis_ci_(lsis, data)
    lsis.prepare_model_states_()
    return lsis

def _load_lsis_ci_(lsis, data):
    ci_sf = [[None for s in range (2)] for i in range (lsis.nfrags)]
    ci_ch = [[[[None,None] for s in range (4)]
              for a in range (lsis.nfrags)]
             for i in range (lsis.nfrags)]
    d = data['ci_sf']
    for i in range (lsis.nfrags):
        di = d[str(i)]
        for s in range (2):
            if str(s) in di:
                ci_sf[i][s] = di[str(s)]
    lsis.ci_spin_flips = ci_sf
    d = data['ci_ch']
    for i in range (lsis.nfrags):
        di = d[str(i)]
        for a in range (lsis.nfrags):
            dia = di[str(a)]
            for s in range (4):
                dias = dia[str(s)]
                for p in range (2):
                    if str(p) in dias:
                        ci_ch[i][a][s][p] = dias[str(p)]
    lsis.ci_charge_hops = ci_ch
    return lsis

def dump_lsis (lsis, chkfile=None, method_key='lsi', mo_coeff=None,
               overwrite_mol=True, keys_config=KEYS_CONFIG_LASSIS,
               keys_saconstr=KEYS_SACONSTR_LASSIS,
               keys_results=KEYS_RESULTS_LASSIS,
               **kwargs):
    las_chkfile.dump_las (lsis._las, chkfile=chkfile, method_key='las', mo_coeff=mo_coeff,
                          ci=lsis._las.ci, overwrite_mol=overwrite_mol)

    if chkfile is None: chkfile = lsis.chkfile
    if not chkfile: return lsis
    if mo_coeff is None: mo_coeff = lsis.mo_coeff
    kwargs['mo_coeff'] = mo_coeff

    data = las_chkfile._dump_las_get_data (lsis, keys_config, keys_saconstr, keys_results,
                                           **kwargs)
    with h5py.File (chkfile, 'a') as fh5:
        chkdata = las_chkfile._dump_las_get_chkdata (lsis, fh5, overwrite_mol, method_key)
        las_chkfile._dump_las_1_(lsis, chkdata, data, mo_coeff)
        _dump_lsis_ci_(lsis, chkdata)
    return lsis

def _dump_lsis_ci_(lsis, chkdata):
    ci_sf = lsis.ci_spin_flips
    d = chkdata.create_group ('ci_sf')
    for i in range (lsis.nfrags):
        di = d.create_group (str (i))
        for s in range (2):
            if ci_sf[i][s] is not None:
                di[str(s)] = ci_sf[i][s]
    ci_ch = lsis.ci_charge_hops
    d = chkdata.create_group ('ci_ch')
    for i in range (lsis.nfrags):
        di = d.create_group (str (i))
        for a in range (lsis.nfrags):
            dia = di.create_group (str (a))
            for s in range (4):
                dias = dia.create_group (str (s))
                for p in range (2):
                    if ci_ch[i][a][s][p] is not None:
                        dias[str(p)] = ci_ch[i][a][s][p]

