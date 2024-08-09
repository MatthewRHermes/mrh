import numpy as np
import itertools

def get_grad_exact(a_idxs, i_idxs, ham, las_rdm1, las_rdm2, las_rdm3, epsilon=0.0):
    g = []
    gen_indices = []
    a_idxs_lst = []
    i_idxs_lst = []
    len_a_idx = len(a_idxs)

    grad_h1t1 = get_grad_h1t1(a_idxs, i_idxs, las_rdm1, ham[1]) 
    grad_h2t1 = get_grad_h2t1(a_idxs, i_idxs, las_rdm2, ham[2])
    grad_h1t2 = get_grad_h1t2(a_idxs, i_idxs, las_rdm2, ham[1])

    print ("t1 grads = ", grad_h1t1, grad_h2t1)
    return

def get_h1e_spin(h1):
    nao = h1.shape[0]
    nso = 2*nao
    h1es = np.zeros((nso,nso))
    h1es[:nao,:nao] = h1
    h1es[nao:,nao:] = h1
    return h1es

def get_h2e_spin(h2):
    nao = h2.shape[0]
    nso = 2*nao
    h2es = np.zeros((nso,nso,nso,nso))
    h2es[:nao,:nao,:nao,:nao] = h2
    h2es[:nao,:nao,nao:,nao:] = h2
    h2es[nao:,nao:,:nao,:nao] = h2
    h2es[nao:,nao:,nao:,nao:] = h2
    return h2es

def get_rdm1_spin(rdm1):
    n = rdm1.shape[0]
    nso = 2*n
    rdm1s = np.zeros((nso,nso))
    rdm1s[:n,:n] = rdm1[0]
    rdm1s[n:,n:] = rdm1[1]
    return rdm1s

def get_rdm2_spin(rdm2):
    n = rdm2.shape[-1]
    nso = 2*n
    rdm2s = np.zeros((nso,nso,nso,nso))
    rdm2s[:n,:n,:n,:n] = rdm2[0] #aa
    rdm2s[:n,:n,n:,n:] = rdm2[1] #ab
    rdm2s[n:,n:,:n,:n] = rdm2[1] #ba
    rdm2s[n:,n:,n:,n:] = rdm2[2] #bb
    return rdm2s

def get_grad_h1t1(a_idxs, i_idxs, las_rdm1, h1):
    a_idxes = np.asarray(a_idxs, dtype=object)
    t1a = [a for b in a_idxes if len(b)==1 for a in b]
    i_idxes = np.asarray(i_idxs, dtype=object)
    t1i = [a for b in i_idxes if len(b)==1 for a in b]
    print ("t1a, t1i = ", t1a, t1i, h1)
    h = get_h1e_spin(h1)
    nso = h.shape[0]
    h1t1 = []

    dm1s = get_rdm1_spin(las_rdm1)

    for u,x in zip(t1a,t1i):
        h1t1s = 0.0
        for p in range(nso):
            h1t1s += 2*h[p,u]*dm1s[p,x]-2*h[p,x]*dm1s[p,u]
        h1t1.append(h1t1s)
    print ("h1t1 = ", h1t1)
    return h1t1

def get_grad_h2t1(a_idxs, i_idxs, las_rdm2, h2):
    a_idxes = np.asarray(a_idxs, dtype=object)
    t1a = [a for b in a_idxes if len(b)==1 for a in b]
    i_idxes = np.asarray(i_idxs, dtype=object)
    t1i = [a for b in i_idxes if len(b)==1 for a in b]

    g = get_h2e_spin(h2)
    nso = g.shape[0]
    print (t1a,t1i,h2)
    rdm2s = get_rdm2_spin(las_rdm2)
    h2t1 = []

    for u,x in zip(t1a,t1i):
        print (u,x)
        grad = 0.0
        for p,q,s in itertools.product(range(nso), repeat=3):
            grad += 0.5*(g[p,u,q,s]*rdm2s[p,x,q,s] - g[p,q,s,u]*rdm2s[p,x,s,q] - g[x,q,p,s]*rdm2s[u,q,p,s] + g[p,q,x,s]*rdm2s[u,q,p,s] - g[p,x,q,s]*rdm2s[p,u,q,s] + g[p,q,s,x]*rdm2s[p,u,s,q] + g[u,q,p,s]*rdm2s[x,q,p,s]- g[p,q,u,s]*rdm2s[x,q,p,s])
            print ("u,x,p,q,s,h2t1 = ", u,x,p,q,s,grad)
        h2t1.append(grad)
    print ("h2t1 = ", h2t1)
    return h2t1

def get_grad_h1t2(a_idxs, i_idxs, las_rdm2, h1):
    a_idxes = np.asarray(a_idxs, dtype=object)
    t1a = [a for b in a_idxes if len(b)==1 for a in b]
    t2a = a_idxes[len(t1a):] 
    i_idxes = np.asarray(i_idxs, dtype=object)
    t1i = [a for b in i_idxes if len(b)==1 for a in b]
    t2i = i_idxes[len(t1i):]

    h = get_h1e_spin(h1)
    nso = h.shape[0]
    rdm2s = get_rdm2_spin(las_rdm2)
    h1t2 = []

    for w,z in zip(t2a,t2i):
        print (w,z)
        u,v = w
        x,y = z
        grad_update = 0.0
        for p in range(nso):
            grad = 2*(h[p,u]*rdm2s[p,x,v,y] - h[p,v]*rdm2s[p,x,u,y] - h[p,x]*rdm2s[u,p,v,y] + h[p,y]*rdm2s[u,p,v,x])
            print ("grad = ", grad)
            grad_update += grad
            print ("u,v,x,y,p,h1t2 = ", u,v,x,y,p,grad_update)
        h1t2.append(grad)
    print ("h1t2 = ", h1t2)
    return h1t2
