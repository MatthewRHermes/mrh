from mrh.my_pyscf.mcscf import chkfile as las_chkfile

KEYS_CONFIG_LASSI = las_chkfile.KEYS_CONFIG_LASSCF + ['nfrags', 'break_symmetry', 'soc', 'opt']
KEYS_SACONSTR_LASSI = las_chkfile.KEYS_SACONSTR_LASSCF
KEYS_RESULTS_LASSI = ['e_states', 'e_roots', 'si', 's2', 'nelec', 'wfnsym', 'rootsym']

def load_lsi_(lsi, chkfile=None, method_key='lsi',
              keys_config=KEYS_CONFIG_LASSI,
              keys_saconstr=KEYS_SACONSTR_LASSI,
              keys_results=KEYS_RESULTS_LASSI):
    lsi._las.load_chk (chkfile=chkfile)
    return las_chkfile.load_las_(lsi, chkfile=chkfile, method_key=method_key,
                                 keys_config=keys_config,
                                 keys_saconstr=keys_saconstr, keys_results=keys_results)

def dump_lsi (lsi, chkfile=None, method_key='lsi', mo_coeff=None, ci=None,
              overwrite_mol=True, keys_config=KEYS_CONFIG_LASSI,
              keys_saconstr=KEYS_SACONSTR_LASSI,
              keys_results=KEYS_RESULTS_LASSI,
              **kwargs):
    lsi._las.dump_chk (chkfile=chkfile)
    return las_chkfile.dump_las (lsi, chkfile=chkfile, method_key=method_key, mo_coeff=mo_coeff,
                                 ci=ci, overwrite_mol=overwrite_mol, keys_config=keys_config,
                                 keys_saconstr=keys_saconstr, keys_results=keys_results, **kwargs)

                                  



