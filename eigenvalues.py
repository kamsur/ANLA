import numpy as np


def gershgorin(A):
    λ_min, λ_max = np.inf,-np.inf
    (m,n)=A.shape
    for i in range(m):
      r=np.sum(np.absolute(A[i]))-np.absolute(A[i][i])
      λ_min=np.minimum(λ_min,A[i][i]-r)
      λ_max=np.maximum(λ_max,A[i][i]+r)
    # todo
    return λ_min, λ_max


def power(A, v0):
    v = np.array(v0.copy())
    λ = 0
    err = []
    while(True):
      w = A@v
      v = w/np.linalg.norm(w,2)
      λ = (v.T)@(A@v)
      e=np.linalg.norm(A@v-λ*v,np.inf)
      err.append(e)
      if e<=10e-13:
        break

    # todo

    return v, λ, err


def inverse(A, v0, μ):
    v = np.array(v0.copy())
    m=A.shape[0]
    λ = 0
    err = []
    while(True):
      w = np.linalg.solve(A-μ*np.identity(m),v)
      v = w/np.linalg.norm(w,2)
      λ = (v.T)@(A@v)
      e=np.linalg.norm(A@v-λ*v,np.inf)
      err.append(e)
      if e<=10e-13:
        break

    # todo

    return v, λ, err


def rayleigh(A, v0):
    v = np.array(v0.copy())
    m=A.shape[0]
    λ = (v.T)@(A@v)
    err = []
    while(True):
      w = np.linalg.solve(A-λ*np.identity(m),v)
      v = w/np.linalg.norm(w,2)
      λ = (v.T)@(A@v)
      e=np.linalg.norm(A@v-λ*v,np.inf)
      err.append(e)
      if e<=10e-13:
        break

    # todo

    return v, λ, err


def randomInput(m):
    #! DO NOT CHANGE THIS FUNCTION !#
    A = np.random.rand(m, m) - 0.5
    A += A.T  # make matrix symmetric
    v0 = np.random.rand(m) - 0.5
    v0 = v0 / np.linalg.norm(v0) # normalize vector
    return A, v0


if __name__ == '__main__':
    # todo
    #A=np.array([[-3.5,-0.5,0],[-0.5,-1.5,0.5],[0,0.5,-1]])
    '''A,v=randomInput(3)
    print("A=",A)
    print("v0=",v)
    eigval, eigvect=np.linalg.eig(A)
    print("eigval=",eigval)
    print("eigvect=",eigvect)'''
    #print(gershgorin(A))
    '''res=power(A, v)
    print("power eigval=",res[1])
    print("power eigvect=",res[0])'''
    '''mu=np.random.choice(eigval)+0.1
    print("mu=",mu)
    res=inverse(A, v, mu)
    print("inverse eigval=",res[1])
    print("inverse eigvect=",res[0])'''
    '''res=rayleigh(A, v)
    print("rayleigh eigval=",res[1])
    print("rayleigh eigvect=",res[0])'''
    pass