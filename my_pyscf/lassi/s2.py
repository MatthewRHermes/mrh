import numpy as np

def gencoup_table (local_smults, global_smult):
    ''' Genealogical coupling table for states of 2S+1 = global_smult given len (local_smults)
    subunits with local 2S+1 = local_smults '''

    nfrags = len (local_smults)
    local_2s = np.array (local_smults) - 1
    assert (np.sum (local_2s) % 2 != global_smult % 2), 'illegal parity'
    assert (np.sum (local_2s)+1 >= global_smult), 'total spin too high'
    left_ceiling = np.cumsum (local_2s)
    right_ceiling = np.empty_like (left_ceiling)
    right_ceiling[:] = global_smult-1
    right_ceiling[1:] += np.cumsum (local_2s[::-1])[:-1]
    right_ceiling = right_ceiling[::-1]
    ceiling = np.minimum (left_ceiling, right_ceiling)

    def find_lowerable_nodes (path):
        path = np.array (path)
        idx = np.zeros (nfrags, dtype=np.bool_)
        idx[1:-1] = True
        idx = idx & (path>1)
        left_min = np.array (path)
        left_min[1:] = path[:-1] - local_2s[1:]
        right_min = np.array (path)
        right_min[:-1] = (path - local_2s)[1:]
        idx = idx & (path > left_min)
        idx = idx & (path > right_min)
        return idx

    current_paths = [tuple(ceiling),]
    all_paths = set ()
    for i in range (np.prod (local_smults)):
        if not len (current_paths): break
        all_paths.update (current_paths)
        next_paths = []
        for path in current_paths:
            idx_lowerable = find_lowerable_nodes (path)
            n_lowerable = np.count_nonzero (idx_lowerable)
            if not n_lowerable: continue
            idx_lowerable = np.diag (idx_lowerable)[idx_lowerable]
            new_paths = np.tile (path, (n_lowerable, 1))
            new_paths[idx_lowerable] -= 2
            next_paths.extend (list (new_paths))
        next_paths = [tuple (next_path) for next_path in next_paths]
        next_paths = list (set (next_paths))
        current_paths = next_paths

    all_paths = np.array (list (all_paths))
    for ix in range (2,nfrags):
        all_paths = all_paths[all_paths[:,-ix].argsort (kind='mergesort')]
    return all_paths


if __name__=='__main__':
    print (gencoup_table ([5,3,5],10))

