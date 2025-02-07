import numpy as np
naux=100
nao=10
ncore=4

bufd=np.random.random((naux,nao))
a=np.einsum('ki,kj->ij',bufd,bufd[:,:ncore])
b=bufd.T@bufd[:,:ncore]
print(np.allclose(a,b))
