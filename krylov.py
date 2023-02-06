import numpy as np
from scipy.linalg import solve_triangular

def cg(A, b, tol=1e-12):
    m = A.shape[0]
    x = np.zeros(m, dtype=A.dtype)
    r_b = [1]
    r=b
    p=r
    for i in range(m):
        A_p=A@p
        alpha=(r.T@r)/(p.T@A_p)
        x+=alpha*p
        r_n=r-alpha*A_p
        r_b.append(np.linalg.norm(r_n,2)/np.linalg.norm(b,2))
        if r_b[-1]<tol:
            r=r_n
            break
        beta=(r_n.T@r_n)/(r.T@r)
        p=r_n+beta*p
        r=r_n

    # todo

    return x, r_b


def arnoldi_n(A, Q, P):
    # n-th step of arnoldi
    m, n = Q.shape
    q = np.zeros(m, dtype=Q.dtype)
    h = np.zeros(n + 1, dtype=A.dtype)
    lhs=solve_triangular(P,A)
    v = lhs@(Q[:,-1])
    for j in range(n):
        h[j] = (Q[:,j].T.conjugate())@v
        v = v-h[j]*Q[:,j]
    h[n] = np.linalg.norm(v,2)
    q = v/h[n] if h[n]!=0 else 0
    # todo

    return h, q


def gmres(A, b, P=np.eye(0), tol=1e-12):
    m = A.shape[0]
    if P.shape != A.shape:
        # default preconditioner P = I
        P = np.eye(m)
    x = np.zeros(m, dtype=b.dtype)
    r_b = [1]
    Q = np.zeros((m,1), dtype=b.dtype)
    rhs=solve_triangular(P,b)
    Q[:,0]=rhs/np.linalg.norm(rhs,2)
    H=np.zeros((1,1), dtype=A.dtype)
    for i in range(m):
        Q_next=Q
        if i!=m-1:
            Hrow_to_be_added = np.zeros(H.shape[1],dtype=A.dtype)
            H=np.vstack ((H, Hrow_to_be_added) )
            Hcolumn_to_be_added,Qcolumn_to_be_added = arnoldi_n(A,Q,P)
            if i==0:
                H[:,-1]=Hcolumn_to_be_added
            else:
                H=np.column_stack((H, Hcolumn_to_be_added))
            Q_next=np.column_stack((Q_next, Qcolumn_to_be_added))
        else:
            Hcolumn_to_be_added,Qcolumn_to_be_added = arnoldi_n(A,Q,P)
            if i==0:
                H[:,-1]=Hcolumn_to_be_added
            else:
                H=np.column_stack((H, Hcolumn_to_be_added))
        lsq_q,lsq_r=np.linalg.qr(H)
        e=np.zeros(H.shape[0])
        e[0]=1
        lsq_w=(lsq_q.T.conjugate())@(np.linalg.norm(rhs,2)*e)
        y=solve_triangular(lsq_r,lsq_w)
        r=H@y-np.linalg.norm(rhs,2)*e
        x=Q@y
        Q=Q_next
        r_b.append(np.linalg.norm(r,2)/np.linalg.norm(rhs,2))
        if r_b[-1]<tol:
            break

    # todo

    return x, r_b
