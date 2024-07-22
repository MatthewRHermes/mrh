import numpy as np

def get_grad_exact(a_idxs, i_idxs, h, las_rdm1, las_rdm2, las_rdm3, epsilon=0.0):
    """
    Compute the exact gradients and relevant indices based on input values.

    Parameters:

    - a_idxs: list of lists of arrays
        unoccupied indices of T-amplitudes

    - i_idxs: list of lists of arrays
        occupied indices of T-amplitudes    

    - h: array-like
        h0, h1, h2 coming from LASSCF calculation

    - las_rdm1: array ncasXncas
        1-RDM from LASSCF calculation

    - las_rdm2: array ncasXncasXncasXncas
        2-RDM from LASSCF calculation

    - epsilon (optional): float, default=0.0
        Threshold value for considering a gradient. If epsilon is 0, all gradients are considered.

    Returns:
    - tuple
        g: list of gradients above the epsilon threshold
        gen_indices: list of indices representing a_idx and i_idx
        a_idxs_lst: list of a_idx values
        i_idxs_lst: list of i_idx values
        len(a_idxs_lst): length of a_idx list
        len(i_idxs_lst): length of i_idx list
    """

    g = []
    gen_indices = []
    a_idxs_lst = []
    i_idxs_lst = []
    len_a_idx = len(a_idxs)
    print ("SV length of a_idx = ", len_a_idx)
    
    grad_h1t1, len_t1a_arrays = get_grad_h1t1(a_idxs, i_idxs, las_rdm1, h)

    print ("SV length of t1a = ", len_t1a_arrays)
    gradients = 0.0
    gradients = (grad_h1t1+get_grad_h1t2(a_idxs, i_idxs, len_t1a_arrays, las_rdm2, h)+get_grad_h2t1(a_idxs, i_idxs, las_rdm2, h)+get_grad_h2t2(a_idxs, i_idxs, len_t1a_arrays, las_rdm2, las_rdm3, h))/2
    
    for i in range(len_a_idx):
        if epsilon == 0.0 or abs(gradients[i]) > epsilon: # Allow all gradients if epsilon is 0, else use the abs gradient condition
            g.append((gradients[i], i))
            a_idx = a_idxs[i]
            i_idx = i_idxs[i]

            gen_indices.append((a_idx, i_idx))
            a_idxs_lst.append(a_idx)
            i_idxs_lst.append(i_idx)
    
    return gradients, g, gen_indices, a_idxs_lst, i_idxs_lst, len(a_idxs_lst), len(i_idxs_lst)

def get_grad_h1t1(a_idxs, i_idxs, las_rdm1, h):
    print ("SV a_idxs = ", a_idxs)
    a_idxes = np.asarray(a_idxs, dtype=object)
    print ("SV a_idxes = ", a_idxes, len(a_idxes))
    t1a_arrays = [a for b in a_idxes if len(b)==1 for a in b]
    i_idxes = np.asarray(i_idxs, dtype=object)
    #print ("SV i_idxes = ", i_idxes, len(i_idxes))
    t1i_arrays = [a for b in i_idxes if len(b)==1 for a in b]

    h1 = h[1]

    nao = h1.shape[0]
    nso = 2*nao
    h1_mat = make_h1e_qiskit(nso,h1)
    #h1_mat = make_h1e_qiskit(nso,h1)
    print ("SV h1_mat = ", h1_mat)
    #rdm1 = np.block([[las_rdm1/2,las_rdm1/2],[las_rdm1/2,las_rdm1/2]])
    rdm1 = np.zeros((nso,nso))
    rdm1 [:nao,:nao] = las_rdm1[0]
    rdm1 [nao:,nao:] = las_rdm1[1]

    print ("SV t1a_arrays = ", len(t1a_arrays))
    #print ("SV t1i_arrays = ", len(t1i_arrays))

    h1_t1 = []

    for u,x in zip(t1a_arrays,t1i_arrays):
        h1t1 = 0.0
        for p in range(nso):
            h1t1 = h1t1 + (2*h1_mat[p,u]*rdm1[p,x])-(2*h1_mat[p,x]*rdm1[p,u])
            #print (u,x,p, " --> ",h1t1)
        h1_t1.append(h1t1)

    h1_t1 = np.asarray(h1_t1)
    h1_t1_rem = np.zeros(len(a_idxes)-len(t1a_arrays))
    h1_t1_full = np.concatenate((h1_t1,h1_t1_rem))
    print ("All h1t1 gradients = ", h1_t1_full)
     
    return h1_t1_full, len(t1a_arrays)

def get_grad_h1t2(a_idxs, i_idxs, len_t1a_arrays, las_rdm2, h):
    h1 = h[1]
    nao = h1.shape[0]
    nso = 2*nao
    len_t2a_arrays = len(a_idxs)-len_t1a_arrays

    a_idxes = np.asarray(a_idxs, dtype=object)
    #print ("SV a_idxes = ", a_idxes[8:])
    t2a_arrays = a_idxes[len_t1a_arrays:]
    i_idxes = np.asarray(i_idxs, dtype=object)
    t2i_arrays = i_idxes[len_t1a_arrays:]
    #print ("SV t2a_arrays = ", t2a_arrays, t2a_arrays[0][0])
    h1_mat = make_h1e_qiskit(nso,h1)
    rdm2 = make_rdm2s_mulliken(nso, las_rdm2)
    print ("SV doubles = ", t2a_arrays, t2i_arrays)

    h1_t2 = []

    for a,i in zip(t2a_arrays, t2i_arrays):
        h1t2 = 0.0
        u,v = a
        w,x = i
        for p in range(nso):
            print ("SV rdm2 check: ", rdm2[p,x,v,w], rdm2[x,p,w,v])
            h1t2 += (2*h1_mat[p,u]*rdm2[p,x,v,w])+(2*h1_mat[p,v]*rdm2[u,x,p,w])-(2*h1_mat[p,x]*rdm2[u,p,v,w])-(2*h1_mat[p,w]*rdm2[u,x,v,p])
            print (u,v,w,x,p ,"--> ",h1t2)
        h1_t2.append(h1t2)
        #print (h1t2)
    h1_t2_rem = np.zeros(len_t1a_arrays)
    h1_t2_full = np.concatenate((h1_t2_rem,h1_t2))
    print ("All h1t2 gradients = ", h1_t2_full)
    return h1_t2_full

def get_grad_h2t1(a_idxs, i_idxs, las_rdm2, h):

    a_idxes = np.asarray(a_idxs, dtype=object)
    #print ("SV a_idxes = ", a_idxes, len(a_idxes))
    t1a_arrays = [a for b in a_idxes if len(b)==1 for a in b]
    i_idxes = np.asarray(i_idxs, dtype=object)
    #print ("SV i_idxes = ", i_idxes, len(i_idxes))
    t1i_arrays = [a for b in i_idxes if len(b)==1 for a in b]
    #print ("SV singles = ", t1a_arrays, t1i_arrays)
    
    h2 = h[2]
    nao = h2.shape[0]
    nso = 2*nao
    h2_mat = make_h2e_mulliken(nso,h2)
    #print ("SV h2_mat = ", h2_mat)
    rdm2 = make_rdm2s_mulliken(nso, las_rdm2)
    h2_t1 = []
    for u,x in zip(t1a_arrays,t1i_arrays):
        print ("SV u,x = ", u,x)
        h2t1 = 0.0
        for p in range(nso):
            for q in range(nso):
                for s in range(nso):
                    h2t1 += (2*h2_mat[p,q,s,u]*(rdm2[p,q,s,x]+rdm2[q,p,s,x])) - (2*h2_mat[p,q,s,x]*(rdm2[p,q,u,s]+rdm2[q,p,u,s]))
                    print (u,x,p,q,s, " --> ",h2t1)
        h2_t1.append(h2t1)

    h2_t1 = np.asarray(h2_t1)
    h2_t1_rem = np.zeros(len(a_idxes)-len(t1a_arrays))
    h2_t1_full = np.concatenate((h2_t1,h2_t1_rem))
    print ("All h2t1 gradients = ", h2_t1_full)
    return h2_t1_full

def get_grad_h2t2(a_idxs, i_idxs, len_t1a_arrays, las_rdm2, las_rdm3, h):

    a_idxes = np.asarray(a_idxs, dtype=object)
    #print ("SV a_idxes = ", a_idxes, len(a_idxes))
    t1a_arrays = [a for b in a_idxes if len(b)==1 for a in b]
    i_idxes = np.asarray(i_idxs, dtype=object)
    #print ("SV i_idxes = ", i_idxes, len(i_idxes))
    t1i_arrays = [a for b in i_idxes if len(b)==1 for a in b]

    len_t2a_arrays = len(a_idxs)-len_t1a_arrays

    a_idxes = np.asarray(a_idxs, dtype=object)
    #print ("SV a_idxes = ", a_idxes[8:])
    t2a_arrays = a_idxes[len_t1a_arrays:]
    i_idxes = np.asarray(i_idxs, dtype=object)
    t2i_arrays = i_idxes[len_t1a_arrays:]

    h2 = h[2]
    nao = h2.shape[0]
    nso = 2*nao
    h2_mat = make_h2e_qiskit(nso,h2)

    #rdm2 = np.tile(las_rdm2, (2, 2, 2, 2))
    #rdm2 =rdm2/2
    rdm2 = make_rdm2s_mulliken(nso,las_rdm2)
    rdm3 = make_rdm3_2nso(nso,las_rdm3)
    #rdm3 = np.tile(las_rdm3, (2, 2, 2, 2, 2, 2))
    #rdm3 =rdm3/2

    #print ("SV h2, rdm2 = ", h2_mat.shape,rdm2.shape)
    h2_t2_2rdm = []
    h2_t2_3rdm = []

    for a,i in zip(t2a_arrays, t2i_arrays):
        u = a[0]
        x = a[1]
        y = i[0]
        v = i[1]
        h2t2_3rdm = 0.0
        for p in range(nso):
            for q in range(nso):
                for s in range(nso):
                    h2t2_3rdm += (4*h2_mat[p,q,u,s]*rdm3[p,v,q,s,x,y])+(4*h2_mat[p,q,x,s]*rdm3[p,y,q,s,u,v])-(4*h2_mat[p,q,v,s]*rdm3[p,u,q,s,y,x])-(4*h2_mat[p,q,y,s]*rdm3[p,x,q,s,v,u])
        h2_t2_3rdm.append(h2t2_3rdm)
    for a,i in zip(t2a_arrays, t2i_arrays):
        u = a[0]
        x = a[1]
        y = i[0]
        v = i[1]
        h2t2_2rdm = 0.0
        for p in range(nso):
            for q in range(nso):
                h2t2_2rdm += (4*h2_mat[p,q,u,x]*rdm2[p,v,q,y])-(4*h2_mat[p,q,v,y]*rdm2[p,u,q,x])
                #print (u,x,v,y,p,q, " --> ",h2t2)
        h2_t2_2rdm.append(h2t2_2rdm)
    
    h2_t2_2rdm = np.asarray(h2_t2_2rdm)
    h2_t2_3rdm = np.asarray(h2_t2_3rdm)
    h2_t2 = h2_t2_2rdm + h2_t2_3rdm
    h2_t2_rem = np.zeros(len_t1a_arrays)
    h2_t2_full = np.concatenate((h2_t2_rem,h2_t2))
    print ("All h2t2 gradients = ",h2_t2_full)
    return h2_t2_full

def make_rdm2_2nso(nso,las_rdm2):
    nao = nso//2
    rdm2 = np.zeros((nso,nso,nso,nso))
    rdm2[:nao,:nao,:nao,:nao] = las_rdm2[0]
    rdm2[:nao,:nao,nao:,nao:] = las_rdm2[1]
    rdm2[nao:,nao:,:nao,:nao] = las_rdm2[1]
    rdm2[nao:,nao:,nao:,nao:] = las_rdm2[2]
    print ("SV 2-RDM in 2ncas basis = ", rdm2)
    return rdm2

def make_rdm3_2nso(nso,las_rdm3):
    nao = nso//2
    rdm3 = np.zeros((nso,nso,nso,nso,nso,nso))
    rdm3[:nao,:nao,:nao,:nao,:nao,:nao] = las_rdm3[0]
    rdm3[:nao,:nao,:nao,:nao,nao:,nao:] = las_rdm3[1]
    rdm3[:nao,:nao,nao:,nao:,:nao,:nao] = las_rdm3[2]
    rdm3[:nao,:nao,nao:,nao:,nao:,nao:] = las_rdm3[1]
    rdm3[nao:,nao:,:nao,:nao,:nao,:nao] = las_rdm3[2]
    rdm3[nao:,nao:,:nao,:nao,nao:,nao:] = las_rdm3[2]
    rdm3[nao:,nao:,nao:,nao:,:nao,:nao] = las_rdm3[1]
    rdm3[nao:,nao:,nao:,nao:,nao:,nao:] = las_rdm3[3]
    return rdm3

def make_h1e_2nso(nso,h1):
    h1_mat = np.zeros((nso,nso))
    nao = nso//2
    h1_mat [:nao,:nao] = h1
    h1_mat [nao:,nao:] = h1
    return h1_mat

def make_h2e_2nso(nso,h2):
    nao = nso//2
    h2e = np.zeros((nso,nso,nso,nso))
    h2e[:nao,:nao,:nao,:nao] = h2
    h2e[:nao,:nao,nao:,nao:] = h2
    h2e[nao:,nao:,:nao,:nao] = h2
    h2e[nao:,nao:,nao:,nao:] = h2
    return h2e

def make_h1e_gpt(nso,h1):
    nao = nso//2
    h1e = np.zeros((nso,nso))
    for p in range(nao):
        for q in range(nao):
            h1e[2*p, 2*q] = h1[p, q]       # αα
            h1e[2*p+1, 2*q+1] = h1[p, q]   # ββ
    return h1e

def make_h1e_qiskit(nso,h1):
    nao = nso//2
    h1e = np.zeros((nso,nso))
    threshold = 1E-12
    for p in range(nso):  # pylint: disable=invalid-name
        for q in range(nso):
            spinp = int(p/nao)
            spinq = int(q/nao)
            if spinp % 2 != spinq % 2:
                continue
            ints = h1
            orbp = int(p % nao)
            orbq = int(q % nao)
            if abs(ints[orbp, orbq]) > threshold:
                h1e[p, q] = ints[orbp, orbq]

    return h1e

def make_rdm1s_qiskit(nso,las_rdm1):
    nao = nso//2
    h1e = np.zeros((nso,nso))
    threshold = 1E-12
    for p in range(nso):
        for q in range(nso):
            spinp = int(p/nao)
            spinq = int(q/nao)
            if spinp % 2 != spinq % 2:
                continue
            ints = las_rdm1
            orbp = int(p % nao)
            orbq = int(q % nao)
            if abs(ints[orbp, orbq]) > threshold:
                h1e[p, q] = ints[orbp, orbq]

    return h1e

def make_h2e_mulliken(nso,h2e):
    n = nso//2
    eri_spinless = np.zeros([nso, nso, nso, nso])
    eri_spinless[:n, :n, :n, :n] = h2e
    eri_spinless[:n, :n, n:, n:] = h2e
    eri_spinless[n:, n:, :n, :n] = h2e
    eri_spinless[n:, n:, n:, n:] = h2e
    return eri_spinless

def make_rdm2s_mulliken(nso,rdm2):
    n = nso//2
    eri_spinless = np.zeros([nso, nso, nso, nso])
    eri_spinless[:n, :n, :n, :n] = rdm2[0]
    eri_spinless[:n, :n, n:, n:] = rdm2[1]
    eri_spinless[n:, n:, :n, :n] = rdm2[1]
    eri_spinless[n:, n:, n:, n:] = rdm2[2]
    print ("2-RDM Dirac = ", eri_spinless)
    return eri_spinless

def make_h2e_qiskit(nso,h2):
    #ints_aa = np.einsum('ijkl->ljik', h2)
    ints_bb = ints_ba = ints_ab = ints_aa = h2
    norbs = nso//2
    threshold=1E-12
    moh2_qubit = np.zeros([nso, nso, nso, nso])
    for p in range(nso):
        for q in range(nso):
            for r in range(nso):
                for s in range(nso):
                    spinp = int(p/norbs)
                    spinq = int(q/norbs)
                    spinr = int(r/norbs)
                    spins = int(s/norbs)
                    if spinp != spins:
                        continue
                    if spinq != spinr:
                        continue
                    if spinp == 0:
                        ints = ints_aa if spinq == 0 else ints_ba
                    else:
                        ints = ints_ab if spinq == 0 else ints_bb
                    orbp = int(p % norbs)
                    orbq = int(q % norbs)
                    orbr = int(r % norbs)
                    orbs = int(s % norbs)
                    if abs(ints[orbp, orbq, orbr, orbs]) > threshold:
                        moh2_qubit[p, q, r, s] = -0.5*ints[orbp, orbq, orbr, orbs]

    return moh2_qubit

def make_rdm2s_qiskit(nso,las_rdm2):
    #ints_aa = np.einsum('ijkl->ljik', las_rdm2[0])
    #ints_bb = np.einsum('ijkl->ljik', las_rdm2[2])
    #ints_ba = ints_ab = np.einsum('ijkl->ljik', las_rdm2[1])
    ints_aa = las_rdm2[0]
    ints_ab = ints_ba = las_rdm2[1]
    ints_bb = las_rdm2[2]
    norbs = nso//2
    threshold=1E-12
    rdm2s = np.zeros([nso, nso, nso, nso])
    for p in range(nso):
        for q in range(nso):
            for r in range(nso):
                for s in range(nso):
                    spinp = int(p/norbs)
                    spinq = int(q/norbs)
                    spinr = int(r/norbs)
                    spins = int(s/norbs)
                    if spinp != spins:
                        continue
                    if spinq != spinr:
                        continue
                    if spinp == 0:
                        ints = ints_aa if spinq == 0 else ints_ba
                    else:
                        ints = ints_ab if spinq == 0 else ints_bb
                    orbp = int(p % norbs)
                    orbq = int(q % norbs)
                    orbr = int(r % norbs)
                    orbs = int(s % norbs)
                    if abs(ints[orbp, orbq, orbr, orbs]) > threshold:
                        rdm2s[p, q, r, s] = ints[orbp, orbq, orbr, orbs]
    return rdm2s
