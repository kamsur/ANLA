import numpy as np

'''A = np.array([[2,7,1],
             [3,-8,-1],
             [1,5,3]])'''

def lu(A):
    # TODO
    U = None
    L = None
    m = A.shape[0]
    U = A.astype(float)
    L = np.identity(m)
    for k in range (m-1):
        for j in range (k+1,m):
            L[j][k] = np.divide(U[j][k],U[k][k])
            U[j][k:m] = U[j][k:m] - np.multiply(L[j][k],U[k][k:m])
    return (L, U)

def maxabs_idx(A):
    # TODO
    i, j = (0, 0)
    max=np.absolute(A[0][0])
    (m,n) = A.shape
    for k in range (m):
        for l in range (n):
             if np.absolute(A[k][l])>max:
                (i,j)=(k,l)
                max=np.absolute(A[k][l])
    return (i, j)


def lu_complete(A):
    # TODO
    U = None
    L = None
    P = None
    Q = None

    return (P, Q, L, U)