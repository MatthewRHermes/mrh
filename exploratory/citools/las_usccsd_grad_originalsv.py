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
    
    grad_h1t1, len_t1a_arrays, h1t1_s = get_grad_h1t1(a_idxs, i_idxs, las_rdm1, h)

    print ("SV length of t1a = ", len_t1a_arrays)
    
    gradients = grad_h1t1+get_grad_h1t2(a_idxs, i_idxs, len_t1a_arrays, las_rdm2, h)[0]+get_grad_h2t1(a_idxs, i_idxs, las_rdm2, h)[0]+get_grad_h2t2(a_idxs, i_idxs, len_t1a_arrays, las_rdm2, las_rdm3, h)[0]
    
    for i in range(len_a_idx):
        if epsilon == 0.0 or abs(gradients[i]) > epsilon: # Allow all gradients if epsilon is 0, else use the abs gradient condition
            g.append((gradients[i], i))
            a_idx = a_idxs[i]
            i_idx = i_idxs[i]

            gen_indices.append((a_idx, i_idx))
            a_idxs_lst.append(a_idx)
            i_idxs_lst.append(i_idx)
    
    gradients_t1 = h1t1_s+get_grad_h2t1(a_idxs, i_idxs, las_rdm2, h)[1]
    g_t1 = []
    for i in range(len_t1a_arrays):
        if epsilon == 0.0 or abs(gradients[i]) > epsilon:
            g_t1.append((gradients_t1[i],i))
    print ("SV g_t1 = ", g_t1)

    return gradients, g, gen_indices, a_idxs_lst, i_idxs_lst, len(a_idxs_lst), len(i_idxs_lst)

def get_grad_h1t1(a_idxs, i_idxs, las_rdm1, h):

     a_idxes = np.asarray(a_idxs)
     #print ("SV a_idxes = ", a_idxes, len(a_idxes))
     t1a_arrays = [a for b in a_idxes if len(b)==1 for a in b]
     i_idxes = np.asarray(i_idxs)
     #print ("SV i_idxes = ", i_idxes, len(i_idxes))
     t1i_arrays = [a for b in i_idxes if len(b)==1 for a in b]

     h1 = h[1]

     nao = h1.shape[0]
     nso = 2*nao
     h1 = h1/2
     h1_mat = np.block([[h1,h1],[h1,h1]])

     rdm1 = np.block([[las_rdm1/2,las_rdm1/2],[las_rdm1/2,las_rdm1/2]])


     print ("SV t1a_arrays = ", len(t1a_arrays))
     #print ("SV t1i_arrays = ", len(t1i_arrays))

     h1_t1 = []

     for u,x in zip(t1a_arrays,t1i_arrays):
         h1t1 = 0.0
         for p in range(nso):
             h1t1 = h1t1 + (h1_mat[p,u]*(rdm1[p,x]-rdm1[x,p]))+(h1_mat[p,x]*(rdm1[p,u]-rdm1[u,p]))
             #print (u,x,p, " --> ",h1t1)
         h1_t1.append(h1t1)

     h1_t1 = np.asarray(h1_t1)
     h1_t1_rem = np.zeros(len(a_idxes)-len(t1a_arrays))
     h1_t1_full = np.concatenate((h1_t1,h1_t1_rem))
     print ("All h1t1 gradients = ", h1_t1_full)
     return h1_t1_full, len(t1a_arrays), np.asarray(h1_t1)

def get_grad_h1t2(a_idxs, i_idxs, len_t1a_arrays, las_rdm2, h):
    h1 = h[1]
    nao = h1.shape[0]
    nso = 2*nao
    len_t2a_arrays = len(a_idxs)-len_t1a_arrays

    a_idxes = np.asarray(a_idxs)
    #print ("SV a_idxes = ", a_idxes[8:])
    t2a_arrays = a_idxes[len_t1a_arrays:]
    i_idxes = np.asarray(i_idxs)
    t2i_arrays = i_idxes[len_t1a_arrays:]
    #print ("SV t2a_arrays = ", t2a_arrays, t2a_arrays[0][0])

    h1 = h1/2
    h1_mat = np.block([[h1,h1],[h1,h1]])

    #rdm2 = np.block([[[[las_rdm2,las_rdm2],[las_rdm2,las_rdm2]],[[las_rdm2,las_rdm2],[las_rdm2,las_rdm2]]],[[[las_rdm2,las_rdm2],[las_rdm2,las_rdm2]],[[las_rdm2,las_rdm2],[las_rdm2,las_rdm2]]]])
    rdm2 = np.tile(las_rdm2, (2, 2, 2, 2))
    rdm2 =rdm2/2
    #print ("SV h1, rdm2 = ", h1_mat.shape,rdm2.shape)
    h1_t2 = []

    for a,i in zip(t2a_arrays, t2i_arrays):
        h1t2 = 0.0
        u = a[0]
        x = a[1]
        y = i[0]
        v = i[1]
        for p in range(nso):
            h1t2 = h1t2 + (h1_mat[p,u]*(rdm2[p,v,x,y]+rdm2[v,p,y,x]))+(h1_mat[p,x]*(rdm2[u,v,p,y]+rdm2[v,u,y,p]))-(h1_mat[v,p]*(rdm2[u,p,x,y]+rdm2[p,u,y,x]))+(h1_mat[p,y]*(rdm2[v,u,p,x]-rdm2[u,v,x,p]))
            #h1t2 = h1t2 + (h1_mat[p,u]*(rdm2[p,x,v,y]+rdm2[v,y,p,x]))+(h1_mat[p,x]*(rdm2[u,p,v,y]-rdm2[v,y,u,p]))-(h1_mat[v,p]*(rdm2[u,x,p,y]-rdm2[p,y,u,x]))+(h1_mat[p,y]*(rdm2[v,p,u,x]-rdm2[u,x,v,p]))
        #print (u,x,y,v,p ,"--> ",h1t2)
        h1_t2.append(h1t2)
        #print (h1t2)
    h1_t2_rem = np.zeros(len_t1a_arrays)
    h1_t2_full = np.concatenate((h1_t2_rem,h1_t2))
    print ("All h1t2 gradients = ", h1_t2_full)
    return h1_t2_full, np.asarray(h1_t2)

def get_grad_h2t1(a_idxs, i_idxs, las_rdm2, h):

    a_idxes = np.asarray(a_idxs)
    #print ("SV a_idxes = ", a_idxes, len(a_idxes))
    t1a_arrays = [a for b in a_idxes if len(b)==1 for a in b]
    i_idxes = np.asarray(i_idxs)
    #print ("SV i_idxes = ", i_idxes, len(i_idxes))
    t1i_arrays = [a for b in i_idxes if len(b)==1 for a in b]
    
    h2 = h[2]
    nao = h2.shape[0]
    nso = 2*nao
    h2_mat = np.block([[[[h2,h2],[h2,h2]],[[h2,h2],[h2,h2]]],[[[h2,h2],[h2,h2]],[[h2,h2],[h2,h2]]]])
    h2_mat = h2_mat/2
    
    rdm2 = np.tile(las_rdm2, (2, 2, 2, 2))
    #rdm2 = np.block([[[[las_rdm2,las_rdm2],[las_rdm2,las_rdm2]],[[las_rdm2,las_rdm2],[las_rdm2,las_rdm2]]],[[[las_rdm2,las_rdm2],[las_rdm2,las_rdm2]],[[las_rdm2,las_rdm2],[las_rdm2,las_rdm2]]]])
    rdm2 =rdm2/2
    #print ("SV h2, rdm2 = ", h2_mat.shape,rdm2.shape)
    h2_t1 = []
     
    for u,x in zip(t1a_arrays,t1i_arrays):
        h2t1 = 0.0
        for p in range(nso):
            for q in range(nso):
                for s in range(nso):
#                    h2t1 += (h2_mat[p,q,u,s]*rdm2[p,q,x,s])+(h2_mat[p,q,s,u]*rdm2[p,q,s,x])-(h2_mat[x,p,q,s]*rdm2[u,p,q,s])-(h2_mat[p,x,q,s]*rdm2[p,u,q,s])-(h2_mat[u,p,q,s]*rdm2[x,p,q,s])-(h2_mat[p,u,q,s]*rdm2[p,x,q,s])+(h2_mat[p,q,x,s]*rdm2[p,q,u,s])+(h2_mat[p,q,s,x]*rdm2[p,q,s,u])
                    h2t1 += (h2_mat[p,q,u,s]*rdm2[p,x,q,s])+(h2_mat[p,q,s,u]*rdm2[p,s,q,x])-(h2_mat[x,p,q,s]*rdm2[u,q,p,s])-(h2_mat[p,x,q,s]*rdm2[p,q,u,s])-(h2_mat[u,p,q,s]*rdm2[x,q,p,s])-(h2_mat[p,u,q,s]*rdm2[p,q,x,s])+(h2_mat[p,q,x,s]*rdm2[p,u,q,s])+(h2_mat[p,q,s,x]*rdm2[p,s,q,u])
                    #print (u,x,p,q,s, " --> ",h2t1)
        h2_t1.append(h2t1)

    h2_t1 = np.asarray(h2_t1)
    h2_t1_rem = np.zeros(len(a_idxes)-len(t1a_arrays))
    h2_t1_full = np.concatenate((h2_t1,h2_t1_rem))
    print ("All h2t1 gradients = ", h2_t1_full)
    return h2_t1_full, np.asarray(h2_t1)

def get_grad_h2t2(a_idxs, i_idxs, len_t1a_arrays, las_rdm2, las_rdm3, h):

    a_idxes = np.asarray(a_idxs)
    #print ("SV a_idxes = ", a_idxes, len(a_idxes))
    t1a_arrays = [a for b in a_idxes if len(b)==1 for a in b]
    i_idxes = np.asarray(i_idxs)
    #print ("SV i_idxes = ", i_idxes, len(i_idxes))
    t1i_arrays = [a for b in i_idxes if len(b)==1 for a in b]

    len_t2a_arrays = len(a_idxs)-len_t1a_arrays

    a_idxes = np.asarray(a_idxs)
    #print ("SV a_idxes = ", a_idxes[8:])
    t2a_arrays = a_idxes[len_t1a_arrays:]
    i_idxes = np.asarray(i_idxs)
    t2i_arrays = i_idxes[len_t1a_arrays:]

    h2 = h[2]
    nao = h2.shape[0]
    nso = 2*nao
    h2_mat = np.block([[[[h2,h2],[h2,h2]],[[h2,h2],[h2,h2]]],[[[h2,h2],[h2,h2]],[[h2,h2],[h2,h2]]]])
    h2_mat = h2_mat/2
    
    rdm2 = np.tile(las_rdm2, (2, 2, 2, 2))
    #rdm2 = np.block([[[[las_rdm2,las_rdm2],[las_rdm2,las_rdm2]],[[las_rdm2,las_rdm2],[las_rdm2,las_rdm2]]],[[[las_rdm2,las_rdm2],[las_rdm2,las_rdm2]],[[las_rdm2,las_rdm2],[las_rdm2,las_rdm2]]]])
    rdm2 =rdm2/2

    rdm3 = np.tile(las_rdm3, (2, 2, 2, 2, 2, 2))
    rdm3 =rdm3/2

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
                    h2t2_3rdm += (h2_mat[p,q,u,s]*(rdm3[p,v,q,s,x,y]-rdm3[p,s,q,v,x,y]))+(h2_mat[p,q,x,s]*(rdm3[p,y,q,s,u,v]-rdm3[p,s,q,y,u,v]))-(h2_mat[v,p,q,s]*(rdm3[u,q,x,y,p,v]-rdm3[u,s,x,y,p,q]))-(h2_mat[y,p,q,s]*(rdm3[u,v,x,q,p,s]-rdm3[u,v,x,s,p,q]))-(h2_mat[p,q,v,s]*(rdm3[p,u,q,s,y,x]-rdm3[p,s,q,u,y,x]))-(h2_mat[p,q,y,s]*(rdm3[p,x,q,s,v,u]-rdm3[p,s,q,x,v,u]))+(h2_mat[u,p,q,s]*(rdm3[v,q,y,x,p,s]-rdm3[v,s,y,x,p,q]))+(h2_mat[x,p,q,s]*(rdm3[v,u,y,q,p,s]-rdm3[v,u,y,s,p,q]))
        h2_t2_3rdm.append(h2t2_3rdm)
    for a,i in zip(t2a_arrays, t2i_arrays):
        u = a[0]
        x = a[1]
        y = i[0]
        v = i[1]
        h2t2_2rdm = 0.0
        for p in range(nso):
            for q in range(nso):
                h2t2_2rdm += (h2_mat[p,q,u,x]*(rdm2[p,v,q,y]-rdm2[p,y,q,v]))-(h2_mat[v,y,p,q]*(rdm2[u,p,x,q]-rdm2[u,q,x,p]))-(h2_mat[p,q,v,y]*(rdm2[p,u,q,x]-rdm2[p,x,q,u]))+(h2_mat[u,v,p,q]*(rdm2[v,p,y,q]-rdm2[v,q,y,p]))
                #print (u,x,v,y,p,q, " --> ",h2t2)
        h2_t2_2rdm.append(h2t2_2rdm)
    
    h2_t2_2rdm = np.asarray(h2_t2_2rdm)
    h2_t2_3rdm = np.asarray(h2_t2_3rdm)
    h2_t2 = h2_t2_2rdm + h2_t2_3rdm
    h2_t2_rem = np.zeros(len_t1a_arrays)
    h2_t2_full = np.concatenate((h2_t2_rem,h2_t2))
    print ("All h2t2 gradients = ",h2_t2_full)
    return h2_t2_full, np.asarray(h2_t2)
