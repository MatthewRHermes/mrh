import h5py
from pyscf.lib import chkfile as chk
from pyscf.lib import H5FileWrap

def has_chk (chkfile, with_record=None):
    if not bool (chkfile): return False
    if h5py.is_hdf5(chkfile):
        if with_record is None:
            return True
        with H5FileWrap(chkfile, 'r+') as fh5:
            if with_record in fh5:
                return True
            else:
                return False
    else:
        return False

def iterate_down (item, f):
    if isinstance (item, list):
        return [iterate_down (i,f) for i in item]
    elif isinstance (item, tuple):
        return (iterate_down (i,f) for i in item)
    elif isinstance (item, dict):
        return {iterate_down (k,f): iterate_down (v,f) for k, v in item.items ()}
    return f (item)

def dump (chkfile, key, value):
    def myfilter (item):
        if item is None:
            return '''__from_None__'''
        elif isinstance (item, int):
            return '{}.__from_int__'.format (item)
        else:
            return item
    return chk.dump (chkfile, key, iterate_down (value, myfilter))

def load (chkfile, key):
    def myfilter (item):
        if (item is None) or (isinstance (item,(str,bytes)) and ('''__from_None__''' in str (item))):
            return None
        elif isinstance (item,(str,bytes)) and ('__from_int__' in str(item)):
            item = str (item)
            if item.startswith ("b'"):
                item = item[2:]
            return int (str (item).split ('.')[0])
        else:
            return item
    return iterate_down (chk.load (chkfile, key), myfilter)

