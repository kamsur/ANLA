import numpy as np

def lu(A):
    # TODO
    m = A.shape[0]
    U = A.astype(float)
    L = np.identity(m).astype(float)
    for k in range (m-1):
        L[k+1:,k]=np.divide(U[k+1:,k],U[k][k])
        U[k+1:,k:]=U[k+1:,k:]-np.outer(L[k+1:,k],U[k,k:])

    return (L, U)

def maxabs_idx(A):
    # TODO
    i,j=np.unravel_index(np.argmax(np.absolute(A)),A.shape)
    return (i, j)


def lu_complete(A):
    # TODO
    A=A.astype(float)
    U = A.copy()
    m=A.shape[0]
    L = np.eye(m).astype(float)
    P = np.eye(m).astype(float)
    Q = np.eye(m).astype(float)
    for k in range(m-1):
        max_i, max_j=np.array(maxabs_idx(U[k:,k:]))+k
        U[[k,max_i],k:]=U[[max_i,k],k:]
        U[:,[k,max_j]]=U[:,[max_j,k]]
        L[[k,max_i],:k]=L[[max_i,k],:k]
        P[[k,max_i],:]=P[[max_i,k],:]
        Q[:,[k,max_j]]=Q[:,[max_j,k]]
        L[k+1:,k]=np.divide(U[k+1:,k],U[k][k])
        U[k+1:,k:]=U[k+1:,k:]-np.outer(L[k+1:,k],U[k,k:])

    return (P, Q, L, U)