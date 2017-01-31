import numpy as np
import matplotlib.pyplot as plt

def Power(M):
    u = np.random.random((M.shape[0],1))
    v = np.zeros((M.shape[0],1))
    end = 0.0;
    i = 0

    while(end != 1.0):
        v = u
        Mu = M*u
        mnM = np.linalg.norm(Mu)
        u = Mu/mnM
        u = u/np.linalg.norm(u)

        end = np.linalg.norm(np.matmul(v.T,u))

        # print i-th iteration
        plt.scatter(i,end)
        print("Iteration " + str(i) + " results in value: " + repr(end))

        i += 1
    return (u)

M = np.matrix([[-2,-2,3],[-10,-1,6],[10,-2,-9]])
print("\nResulting Eigenvector:\n" + str(Power(M)))

plt.show()
