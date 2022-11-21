import numpy as np
import math

def sign(x):
    abs_val=math.sqrt(np.real(x)**2+np.imag(x)**2)
    return x/abs_val if abs_val!=0 else 1

def implicit_qr(A):
    R = A.astype(complex)
    (m,n)=R.shape
    W=np.zeros((m,n)).astype(complex)
    for i in range(n):
        temp_x=R[i:,i]
        e=np.zeros(m-i)
        e[0]=1
        sign_vk=sign(temp_x[0])
        vk = sign_vk*np.linalg.norm(temp_x)*e+temp_x
        vk = vk/np.linalg.norm(vk) if np.linalg.norm(vk)!=0 else 1
        y = np.identity(temp_x.size) - (2*np.outer(vk, vk.T.conjugate()))
        R[i:,i:] = np.matmul(y,R[i:,i:])
        W[i:, i] = vk

    #TODO
    return (W, R)

def form_q(W):
    (m,n)=W.shape
    Q=None
    P = np.identity(m).astype(complex)
    for i in range(m):
        for j in range(n):
            P[j:,i] = P[j:,i]-((np.outer(W[j:,j],(W[j:,j]).T.conjugate()))*2)@P[j:,i]
    Q = P.T.conjugate()
    #TODO
    return Q