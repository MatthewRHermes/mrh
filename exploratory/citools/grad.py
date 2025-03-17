import numpy as np
from itertools import permutations

def get_grad_exact(a_idxs, i_idxs, ham, las_rdm1, las_rdm2, las_rdm3, epsilon=0.0):
    """
    Calculates the gradients for all parameters

    Arguments:
    a_idxs (list): list of singles and doubles a indices  
    i_idxs (lsit): list of singles and doubles i indices
    ham (list): list of h0, h1, h2
    las_rdm1 (array): spin-separated 1-RDM in spatial basis
    las_rdm2 (array): spin-separated 2-RDM in spatial basis
    las_rdm3 (array): spin-separated 3-RDM in spatial basis
    epsilon

    Returns:
    gradients (array): all gradients
    gen_indices (list): all combinations i_idxs, a_idxs
    """
    gen_indices = []

    grad_h1t1 = get_grad_h1t1(a_idxs, i_idxs, las_rdm1, ham[1])#*0.5
    grad_h2t1 = get_grad_h2t1(a_idxs, i_idxs, las_rdm2, ham[2])#*0.5
    grad_h1t2 = get_grad_h1t2(a_idxs, i_idxs, las_rdm2, ham[1])#*0.5
    grad_h2t2 = get_grad_h2t2(a_idxs, i_idxs, las_rdm2, las_rdm3, ham[2])#*0.5

    gradients =  np.concatenate((grad_h1t1+grad_h2t1,grad_h1t2+grad_h2t2))
    gen_indices = list(zip(i_idxs, a_idxs))
    np.save('all_grads.npy', gradients)
    
    return gradients, gen_indices

def grad_select(all_gradients, all_gen_indices, a_idxs, i_idxs, epsilon):
    g = []
    gen_indices = []
    a_idxs_lst = []
    i_idxs_lst = []
    len_a_idx = len(a_idxs)

    for i in range(len_a_idx):
        if epsilon == 0.0 or abs(all_gradients[i]) > epsilon:
            g.append((all_gradients[i], i))
            a_idx = a_idxs[i]
            i_idx = i_idxs[i]
            gen_indices.append((a_idx, i_idx))
            a_idxs_lst.append(a_idx)
            i_idxs_lst.append(i_idx)

    return g, gen_indices, a_idxs_lst, i_idxs_lst, len(a_idxs_lst), len(i_idxs_lst)

def get_h1e_spin(h1):
    n = h1.shape[1]
    nso = 2*n
    h1es = np.zeros((nso,nso))
    h1es[:n,:n] = h1
    h1es[n:,n:] = h1
    return h1es

def get_eri_spin(h2):
    n = h2.shape[1]
    nso = 2*n
    eris = np.zeros((nso,nso,nso,nso))
    eris[:n,:n,:n,:n] = h2
    eris[:n,:n,n:,n:] = h2
    eris[n:,n:,:n,:n] = h2
    eris[n:,n:,n:,n:] = h2
    return eris

def get_rdm1_spin(rdm1):
    n = rdm1.shape[1]
    nso = 2*n
    rdm1s = np.zeros((nso,nso))
    rdm1s[:n,:n] = rdm1[0] #a
    rdm1s[n:,n:] = rdm1[1] #b
    return rdm1s

def get_rdm2_spin(rdm2):
    n = rdm2.shape[1]
    nso = 2*n
    rdm2s = np.zeros((nso,nso,nso,nso))
    rdm2s[:n,:n,:n,:n] = rdm2[0]/2 #aa
    rdm2s[:n,:n,n:,n:] = rdm2[1] #ab
    rdm2s[n:,n:,:n,:n] = rdm2[1].transpose(2,3,0,1) #ba
    rdm2s[n:,n:,n:,n:] = rdm2[2]/2 #bb
    rdm2s = rdm2s - rdm2s.transpose (0,3,2,1) # antisymmetrize
    return rdm2s

def get_rdm3_spin(rdm3):
    n = rdm3.shape[1]
    nso = 2*n
    rdm3s = np.zeros((nso,nso,nso,nso,nso,nso))
    rdm3s[:n,:n,:n,:n,:n,:n] = rdm3[0]/6 #aaa
    rdm3s[:n,:n,:n,:n,n:,n:] = rdm3[1]/2 #aab
    rdm3s[:n,:n,n:,n:,n:,n:] = rdm3[2]/2 #abb
    rdm3s[n:,n:,n:,n:,n:,n:] = rdm3[3]/6 #bbb
    rdm3s[:n,:n,n:,n:,:n,:n] = rdm3[1].transpose(0,1,4,5,2,3)/2 #aba
    rdm3s[n:,n:,:n,:n,:n,:n] = rdm3[1].transpose(4,5,0,1,2,3)/2 #baa
    rdm3s[n:,n:,:n,:n,n:,n:] = rdm3[2].transpose(2,3,0,1,4,5)/2 #bab
    rdm3s[n:,n:,n:,n:,:n,:n] = rdm3[2].transpose(2,3,4,5,0,1)/2 #bba

    rdm3s = rdm3s - rdm3s.transpose(0,3,2,1,4,5) + rdm3s.transpose(0,3,2,5,4,1) - rdm3s.transpose(0,5,2,3,4,1) + rdm3s.transpose(0,5,2,1,4,3) - rdm3s.transpose(0,1,2,5,4,3) # antisymmetrize

    return rdm3s

def get_grad_h1t1(a_idxs, i_idxs, las_rdm1, h1):
    a_idxes = np.asarray(a_idxs, dtype=object)
    t1a = [a for b in a_idxes if len(b)==1 for a in b]
    i_idxes = np.asarray(i_idxs, dtype=object)
    t1i = [a for b in i_idxes if len(b)==1 for a in b]
    #print ("t1a, t1i = ", t1a, t1i, h1)
    print (a_idxes, len(a_idxes))
    #print (i_idxes)

    h = get_h1e_spin(h1)
    #print ("h1 = ", h) 
    #print ("gamma = ", get_rdm1_spin(las_rdm1))
    h1t1s = []

    d = get_rdm1_spin(las_rdm1)

    term1 = np.einsum('up,px->ux', h,d)
    term2 = np.einsum('xp,pu->ux', h,d)
    sum_h1t1 = 2.0*(term1-term2)

    #print ("einsum matrix of ux = ", sum_h1t1)
    
    for u,x in zip(t1a,t1i):
        #print ("indices = ",x,"->", u)
        h1t1s.append(sum_h1t1[u,x])
    
    print ("h1t1 = ", h1t1s)
    return np.array(h1t1s)

def get_grad_h2t1(a_idxs, i_idxs, las_rdm2, h2):
    a_idxes = np.asarray(a_idxs, dtype=object)
    t1a = [a for b in a_idxes if len(b)==1 for a in b]
    i_idxes = np.asarray(i_idxs, dtype=object)
    t1i = [a for b in i_idxes if len(b)==1 for a in b]

    g = get_eri_spin(h2)
    d = get_rdm2_spin(las_rdm2)
    h2t1s = []

    gt = g.transpose(0,3,2,1)
    w = g - gt

    term1 = np.einsum('prqu,prqx->ux', w, d)
    term2 = np.einsum('prqx,prqu->ux', w, d)
    sum_h2t1 = term1 - term2

    #print ("einsum matrix of ux = ", sum_h2t1)

    for u,x in zip(t1a,t1i):
        #print ("indices = ",x,"->", u)
        h2t1s.append(sum_h2t1[u,x])
    print ("h2t1 = ", h2t1s)
    return np.array(h2t1s)

def get_grad_h1t2(a_idxs, i_idxs, las_rdm2, h1):
    a_idxes = np.asarray(a_idxs, dtype=object)
    t1a = [a for b in a_idxes if len(b)==1 for a in b]
    t2a = a_idxes[len(t1a):] 
    i_idxes = np.asarray(i_idxs, dtype=object)
    t1i = [a for b in i_idxes if len(b)==1 for a in b]
    t2i = i_idxes[len(t1i):]

    h = get_h1e_spin(h1)
    d = get_rdm2_spin(las_rdm2)
    h1t2s = []

    term1 = np.einsum('pu,xpyv->uxvy',h,d)
    term2 = np.einsum('pv,xpyu->uxvy',h,d)
    term3 = np.einsum('xp,puyv->uxvy',h,d)
    term4 = np.einsum('yp,puxv->uxvy',h,d)

    sum_h1t2 = 2*(term1-term2-term3+term4)
    #print ("einsum matrix of uxvy = ", sum_h1t2)

    for b,z in zip(t2a,t2i):
        u,v = b
        x,y = z
        #print ("indices = ", x,y,"->",u,v, sum_h1t2[u,x,v,y])
        h1t2s.append(sum_h1t2[u,x,v,y])
    
    print ("h1t2 = ", h1t2s)
    return np.array(h1t2s)

def get_grad_h2t2(a_idxs, i_idxs, las_rdm2, las_rdm3, h2):
    a_idxes = np.asarray(a_idxs, dtype=object)
    t1a = [a for b in a_idxes if len(b)==1 for a in b]
    t2a = a_idxes[len(t1a):]
    i_idxes = np.asarray(i_idxs, dtype=object)
    t1i = [a for b in i_idxes if len(b)==1 for a in b]
    t2i = i_idxes[len(t1i):]

    g = get_eri_spin(h2)
    d2 = get_rdm2_spin(las_rdm2)
    d3 = get_rdm3_spin(las_rdm3)
    gt = g.transpose(0,3,2,1)
    w = g - gt

    h2t2s = []
    
    # 2-RDM terms
    term1 = np.einsum('puqv,xpyq->uxvy',w,d2)
    term2 = np.einsum('pxqy,upvq->uxvy',w,d2)
    sum_h2t2_2rdm = term1 - term2

    #3-RDM terms
    term3 = np.einsum('prqu,rpxqyv->uxvy',w,d3)
    term4 = np.einsum('prqv,rpxqyu->uxvy',w,d3)
    term5 = np.einsum('prqx,rpuqvy->uxvy',w,d3)
    term6 = np.einsum('prqy,rpuqvx->uxvy',w,d3)
    sum_h2t2_3rdm = term3 - term4 - term5 + term6

    sum_h2t2 = sum_h2t2_2rdm + sum_h2t2_3rdm

    #print ("einsum matrix of uxvy = ", sum_h2t2)

    for b,z in zip(t2a,t2i):
        u,v = b
        x,y = z
        #print ("indices = ", x,y,"->",u,v, sum_h2t2[u,x,v,y])
        h2t2s.append(sum_h2t2[u,x,v,y])
    print ("h2t2 = ", h2t2s)

    return np.array(h2t2s)
