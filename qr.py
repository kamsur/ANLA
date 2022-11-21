import numpy as np
import math

def sign(x):
    abs_val=math.sqrt(np.real(x)**2+np.imag(x)**2)
    return x/abs_val if abs_val!=0 else 1
'''
def v_k(x):
    if len(x.shape)>0:
        sign_vk=sign(x[0])
    else:
        sign_vk=sign(x)
    e = np.zeros(x.shape)
    e[0]=1
    vk=sign_vk*np.linalg.norm(x,2)*e+x
    return vk/np.linalg.norm(vk,2) if np.linalg.norm(vk,2)!=0 else vk
'''
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
            '''
            e=np.zeros(m-j)
            e[0]=1
            y=e-((np.outer(W[j:,j],np.conj(W[j:,j]).T))*2)
            P[j:,i] =np.matmul(y,P[j:,i])
            '''
            P[j:,i] = P[j:,i]-((np.outer(W[j:,j],(W[j:,j]).T.conjugate()))*2)@P[j:,i]
    Q = P.T.conjugate()
    #TODO
    return Q