import numpy as np
import matplotlib.pyplot as pl

import eigenvalues as ev

def convergence_comparison():
    # fix seed for reproducibility
    np.random.seed(0x2BAD)

    # set matrix dimensions
    m = 2
    # generate random values for A, v0
    A, v0 = ev.randomInput(m)
    eigval, eigvect=np.linalg.eig(A)
    eigmax=np.argmax(np.absolute(eigval))
    v0=eigvect[:,eigmax]+0.2
    print("A=",A)
    print("v0=",v0)
    print("eigval=",eigval)
    print("eigvect=",eigvect)
    # compute estimate for dominant eigenvalue
    λ_min, λ_max = ev.gershgorin(A)
    μ_gershgorin = λ_min if abs(λ_min) > λ_max else λ_max
    # generate random value for μ
    μ_rand = (np.random.rand(1) - 0.5) * np.sqrt(m) * 6
    print("μ_gershgorin=",μ_gershgorin)
    print("μ_rand=",μ_rand)
    # run all algorithms
    _, λp, conv_p = ev.power(A, v0)
    print("vp=0",_)
    _, λir, conv_ir = ev.inverse(A, v0, μ_rand)
    print("vir=0",_)
    _, λig, conv_ig = ev.inverse(A, v0, μ_gershgorin)
    print("vig=0",_)
    _, λr, conv_r = ev.rayleigh(A, v0)
    print("vr=0",_)


    # print results
    print(f'algorithm           |   λ')
    print('--------------------------------------------')
    print(f'power               | {λp}')
    print(f'inverse (random μ)  | {λir}')
    print(f'inverse (Gershgorin)| {λig}')
    print(f'Rayleigh quotient   | {λr}')

    # plot convergence
    pl.semilogy(range(len(conv_p)), conv_p, ".-", label="power iteration")
    pl.semilogy(range(len(conv_ir)), conv_ir, ".-", label="inverse iteration (random μ)")
    pl.semilogy(range(len(conv_ig)), conv_ig, ".-", label="inverse iteration (Gershgorin μ)")
    pl.semilogy(range(len(conv_r)), conv_r, ".-", label="Rayleigh iteration")
    pl.xlabel("iterations")
    pl.ylabel("error")
    pl.legend()
    pl.show()


if __name__ == "__main__":
    convergence_comparison()