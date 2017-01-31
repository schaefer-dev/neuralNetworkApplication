import numpy as np

def Power(M):
    u = np.random.random((M.shape[0],1))
    u1 = np.zeros((M.shape[0],1))
    end=0;
    i = 0
    while(np.array_equal(u,u1)!=True):
        u1 = u
        Mu = M*u
        mnM = np.linalg.norm(Mu)
        u = Mu/mnM
        u = u/np.linalg.norm(u)
    return (u)

M = np.matrix([[-2,-2,3],[-10,-1,6],[10,-2,-9]])
print(Power(M))