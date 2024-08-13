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

def get_eri_spin(h2):
    nao = h2.shape[0]
    nso = 2*nao
    eris = np.zeros((nso,nso,nso,nso))
    eris[:nao,:nao,:nao,:nao] = h2
    eris[:nao,:nao,nao:,nao:] = h2
    eris[nao:,nao:,:nao,:nao] = h2
    eris[nao:,nao:,nao:,nao:] = h2
    return eris

def get_rdm1_spin(rdm1):
    n = rdm1.shape[0]
    nso = 2*n
    rdm1s = np.zeros((nso,nso))
    rdm1s[:n,:n] = rdm1[0] #a
    rdm1s[n:,n:] = rdm1[1] #b
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
        grad = 0.0
        for p in range(nso):
            h1t1s = 2*h[p,u]*dm1s[p,x]-2*h[p,x]*dm1s[p,u]
            grad += h1t1s

            print ("u,x,p, h1t1s, grad = ", u,x,p,h1t1s, grad)
        h1t1.append(grad)
    print ("h1t1 = ", h1t1)
    return h1t1

def get_grad_h2t1(a_idxs, i_idxs, las_rdm2, h2):
    a_idxes = np.asarray(a_idxs, dtype=object)
    t1a = [a for b in a_idxes if len(b)==1 for a in b]
    i_idxes = np.asarray(i_idxs, dtype=object)
    t1i = [a for b in i_idxes if len(b)==1 for a in b]

    g = get_eri_spin(h2)
    nso = g.shape[0]
    #print (t1a,t1i,h2)
    #print (g)
    d = get_rdm2_spin(las_rdm2)
    h2t1s = []
    w = g - g.transpose(0,3,2,1)

    term1 = np.einsum('upqr,xpqr->ux', w, d)
    term2 = np.einsum('xpqr,upqr->ux', w, d)
    sum_h2t1 = term1-term2
    print ("einsum matrix of ux = ", term1, term2, sum_h2t1)
    
    for u,x in zip(t1a,t1i):
        h2t1 = 0.0
        for p,q,r in itertools.product(range(nso), repeat=3):
            #grad = g[p,u,q,s]*rdm2s[p,x,q,s] - g[p,q,s,u]*rdm2s[p,x,s,q] - g[p,x,q,s]*rdm2s[p,u,q,s] + g[p,q,s,x]*rdm2s[p,u,s,q]
            grad = w[u,p,q,r]*d[x,p,q,r]-w[x,p,q,r]*d[u,p,q,r]
            h2t1 +=grad
            #print ("u,x,p,q,r,grad,h2t1 = ", u,x,p,q,r,grad,h2t1)
        h2t1s.append(h2t1)
    print ("h2t1 = ", h2t1s)
    
    return h2t1s

def get_grad_h1t2(a_idxs, i_idxs, las_rdm2, h1):
    a_idxes = np.asarray(a_idxs, dtype=object)
    t1a = [a for b in a_idxes if len(b)==1 for a in b]
    t2a = a_idxes[len(t1a):] 
    i_idxes = np.asarray(i_idxs, dtype=object)
    t1i = [a for b in i_idxes if len(b)==1 for a in b]
    t2i = i_idxes[len(t1i):]

    h = get_h1e_spin(h1)
    nso = h.shape[0]
    d = get_rdm2_spin(las_rdm2)
    h1t2s = []
    #print (las_rdm2)
    #print (rdm2s)

    term1 = np.einsum('up,pxvy->uvxy',h,d)
    term2 = np.einsum('vp,pxuy->uvxy',h,d)
    term3 = np.einsum('xp,puyv->uvxy',h,d)
    term4 = np.einsum('yp,puxv->uvxy',h,d)
    sum_h1t2 = 2*(term1-term2-term3+term4)
    print ("einsum matrix of uvxy = ", sum_h1t2)
    print (sum_h1t2[2,3,0,0], sum_h1t2[2,1,0,2])

    for b,z in zip(t2a,t2i):
        print (b,z)
        u,v = b
        x,y = z
        h1t2 = 0.0
        for p in range(nso):
            term = 2*(h[p,u]*d[x,p,y,v]-h[p,v]*d[p,x,u,y]-h[p,x]*d[u,p,v,y]+h[p,y]*d[u,p,v,x])
            h1t2 += term
            print ("u,v,x,y,p,h1t2 = ", u,v,x,y,p, term, h1t2)
        h1t2s.append(h1t2)
    print ("h1t2 = ", h1t2s)
    return h1t2s
