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
    #h1_mat = make_h1e_qiskit(nso,h1)
    h1_mat = make_h1e_2nso(nso,h1)
    print ("SV h1_mat = ", h1_mat)
    rdm1 = np.zeros((nso,nso))
    rdm1 [:nao,:nao] = las_rdm1[0]
    rdm1 [nao:,nao:] = las_rdm1[1]

    #print ("SV t1a_arrays = ", len(t1a_arrays))
    #print ("SV t1i_arrays = ", len(t1i_arrays))

    h1_t1 = []

    for u,x in zip(t1a_arrays,t1i_arrays):
        print ("u,x from h1t1 = ", u,x)
        h1t1 = 0.0
        for p in range(nso):
            h1t1 = h1t1 + (2*h1_mat[p,u]*rdm1[p,x])-(2*h1_mat[p,x]*rdm1[p,u])
            print (u,x,p, " --> ",h1t1)
        h1_t1.append(h1t1)
    print ("conversion check = ", h1_t1)
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
    h1_mat = make_h1e_2nso(nso,h1)
    rdm2 = make_rdm2s_mulliken(nso, las_rdm2)
    #print ("SV doubles = ", t2a_arrays, t2i_arrays)

    h1_t2 = []
    h1t2 = 0.0
    for a,i in zip(t2a_arrays, t2i_arrays):
        h1t2 = 0.0
        u,v = a
        x,y = i
        for p in range(nso):
            term1 = 2*h1_mat[p,u]*rdm2[v,p,y,x]
            term2 = 2*h1_mat[p,v]*rdm2[u,p,y,x]
            term3 = 2*h1_mat[p,y]*rdm2[x,p,v,u]
            term4 = 2*h1_mat[p,x]*rdm2[y,p,v,u]
            h1t2 += term1 - term2 + term3 - term4
            print ("Terms 1,2,3,4 = ", term1, term2, term3, term4, h1t2)
            #print (u,v,x,y,p ,"--> ",h1t2)
        h1_t2.append(h1t2)
        print (h1t2)
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
    g = make_h2e_mulliken(nso,h2)
    #print ("SV h2_mat = ", h2_mat)
    rdm2 = make_rdm2s_mulliken(nso, las_rdm2)
    gamma = rdm2
    h2_t1 = []
    
    for u,x in zip(t1a_arrays,t1i_arrays):
        print ("SV u,x = ", u,x)
        h2t1 = 0.0
        for p in range (nso):
            for q in range (nso):
                for s in range (nso):
                    h2t1 += 0.5*(g[p,u,q,s]*rdm2[p,x,q,s] - g[p,q,s,u]*rdm2[p,x,s,q] - g[x,q,p,s]*rdm2[u,q,p,s] + g[p,q,x,s]*rdm2[u,q,p,s] - g[p,x,q,s]*rdm2[p,u,q,s] + g[p,q,s,x]*rdm2[p,u,s,q] + g[u,q,p,s]*rdm2[x,q,p,s] - g[p,q,u,s]*rdm2[x,q,p,s])
                    #print ("ft, st = ", ft,st)
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
    g = make_h2e_mulliken(nso,h2)

    rdm2 = make_rdm2s_mulliken(nso,las_rdm2)
    rdm3 = make_rdm3_2nso(nso,las_rdm3)
    
    h2_t2_2rdm = []
    h2_t2_3rdm = []

    for a,i in zip(t2a_arrays, t2i_arrays):
        u = a[0]
        v = a[1]
        x = i[0]
        y = i[1]
        h2t2_3rdm = 0.0
        for p in range(nso):
            for q in range(nso):
                for s in range(nso):
                    h2t2_3rdm += g[p,u,q,s]*rdm3[p,q,v,x,s,y] + g[p,v,q,s]*rdm3[p,q,u,y,s,x] + g[p,q,s,u]*rdm3[p,s,v,q,x,y] + g[p,q,s,v]*rdm3[p,s,u,q,y,x] - g[x,q,p,s]*rdm3[u,v,p,q,y,s] - g[p,q,x,s]*rdm3[u,v,p,s,y,q] - g[y,q,p,s]*rdm3[u,v,p,x,q,s] - g[p,q,y,s]*rdm3[u,v,p,x,s,q] - g[p,x,q,s]*rdm3[p,q,y,u,s,v] - g[p,y,q,s]*rdm3[p,q,x,v,s,u] - g[p,q,s,x]*rdm3[p,s,y,q,u,v] - g[p,q,s,y]*rdm3[p,s,x,q,v,u] + g[u,q,p,s]*rdm3[x,y,p,q,v,s] + g[p,q,u,s]*rdm3[x,y,p,s,v,q] + g[v,q,p,s]*rdm3[x,y,p,u,q,s] + g[p,q,v,s]*rdm3[x,y,p,u,s,q]
        h2_t2_3rdm.append(h2t2_3rdm)
    print ("h2t2 3-RDM part = ", h2_t2_3rdm)
    for a,i in zip(t2a_arrays, t2i_arrays):
        u = a[0]
        v = a[1]
        x = i[0]
        y = i[1]
        h2t2_2rdm = 0.0
        for p in range(nso):
            for q in range(nso):
                h2t2_2rdm += g[p,u,q,v]*rdm2[p,q,x,y] + g[p,v,q,u]*rdm2[p,q,y,x] - g[x,q,y,p]*rdm2[u,v,q,p] - g[y,q,x,p]*rdm2[u,v,p,q] - g[p,q,x,y]*rdm2[p,q,u,v] - g[p,y,q,x]*rdm2[p,q,v,u] + g[u,q,v,p]*rdm2[x,y,q,p] + g[v,q,u,p]*rdm2[x,y,p,q]
                #print (u,x,v,y,p,q, " --> ",h2t2)
        h2_t2_2rdm.append(h2t2_2rdm)
    print ("h2t2 2-RDM part = ", h2_t2_2rdm)
    h2_t2_2rdm = np.asarray(h2_t2_2rdm)/2
    h2_t2_3rdm = np.asarray(h2_t2_3rdm)/2
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
    #print ("SV 2-RDM in 2ncas basis = ", rdm2)
    return rdm2

def make_rdm3_2nso(nso,las_rdm3):
    nao = nso//2
    rdm3 = np.zeros((nso,nso,nso,nso,nso,nso))
    rdm3[:nao, :nao, :nao, :nao, :nao, :nao] = las_rdm3[0] # aaa
    rdm3[:nao, :nao, :nao, :nao, nao:, nao:] = las_rdm3[1] # aab
    rdm3[:nao, :nao, nao:, nao:, :nao, :nao] = las_rdm3[1] # aba
    rdm3[:nao, :nao, nao:, nao:, nao:, nao:] = las_rdm3[2] # abb
    rdm3[nao:, nao:, :nao, :nao, :nao, :nao] = las_rdm3[1] # baa
    rdm3[nao:, nao:, :nao, :nao, nao:, nao:] = las_rdm3[2] # bab
    rdm3[nao:, nao:, nao:, nao:, :nao, :nao] = las_rdm3[2] # bba
    rdm3[nao:, nao:, nao:, nao:, nao:, nao:] = las_rdm3[3] # bbb
    #print ("SV RDM3 = ", rdm3)
    return rdm3

def make_h1e_2nso(nso,h1):
    h1_mat = np.zeros((nso,nso))
    nao = nso//2
    h1_mat [:nao,:nao] = h1
    h1_mat [nao:,nao:] = h1
    return h1_mat

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
    eri_spinless[:n, :n, :n, :n] = rdm2[0] # aa
    eri_spinless[:n, :n, n:, n:] = rdm2[1] # ab
    eri_spinless[n:, n:, :n, :n] = rdm2[1] # ba
    eri_spinless[n:, n:, n:, n:] = rdm2[2] # bb
    return eri_spinless
