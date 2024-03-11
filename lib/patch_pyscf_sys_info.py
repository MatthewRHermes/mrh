import os
import pyscf.lib.misc

if not hasattr (pyscf.lib.misc, 'mrh_patched'):
    from pyscf.lib.misc import format_sys_info as pyscf_format_sys_info
    from pyscf.lib.misc import repo_info
    
    def format_sys_info ():
        result = pyscf_format_sys_info ()
        mrh_info = repo_info (os.path.join (__file__, '..'))
        result.append (f'mrh path  {mrh_info["path"]}')
        if 'git' in mrh_info:
            result.append (mrh_info['git'])
        return result
    
    
    pyscf.lib.misc.format_sys_info = format_sys_info
    pyscf.lib.misc.mrh_patched = True


