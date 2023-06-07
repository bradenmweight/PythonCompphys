import math as m
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

def Gaussian(alpha, r):
    #return np.exp(-r**2 / 2 / alpha**2)
    return np.exp(-alpha * r**2)

alpha = np.array([13.00773, 1.962079, 0.444529, 0.1219492]) # from 3.27
#alpha=(14, 2, 0.5, 0.1) # try something different here to see how it affects the results

NBASIS = len(alpha)
#NBASIS = 3
#alpha = 1/np.linspace( 0.1, 10, NBASIS )

NPTS = 100
f = np.zeros((NPTS, NBASIS))
r = np.linspace(0, 10, NPTS)
for pt in range(NPTS):
    for a in range(NBASIS):
        f[pt, a] = Gaussian(alpha[a], r[pt])

for a in range(NBASIS):
    plt.plot(r[:], f[:,a], label=f"Basis {a}")
plt.savefig("Basis.jpg", dpi=300)
plt.clf()

# Integrals are very simple, the gaussians are centered in the same point (the nucleus), see eqn 3.29:

def Overlap(alpha, ind1, ind2):
    #### Analytic ####
    return np.sqrt( np.pi )**3 / np.sqrt( alpha[ind1] + alpha[ind2] )**3

    #### Numerical ####
    #RGRID = np.linspace(0,10,10000)
    #dr    = RGRID[1] - RGRID[0]
    #S00 = np.sum( RGRID **2 * Gaussian(alpha[ind1], RGRID)**2 ) * dr * 4 * np.pi
    #S11 = np.sum( RGRID **2 * Gaussian(alpha[ind2], RGRID)**2 ) * dr * 4 * np.pi
    #O   = np.sum( RGRID **2 * Gaussian(alpha[ind1], RGRID) * Gaussian(alpha[ind2], RGRID) ) * dr * 4 * np.pi
    #print( round(O,2), round(m.pow(m.pi/(alpha[ind1] + alpha[ind2]), 3./2.),2) )
    return O

def Kinetic(alpha, ind1, ind2):
    return 3*alpha[ind1]*alpha[ind2]*m.pow(m.pi, 3./2.)/m.pow(alpha[ind1] + alpha[ind2],5./2.)

def Coulomb(alpha, ind1, ind2):
    return -2 * m.pi / (alpha[ind1] + alpha[ind2])

def Wavefunction(alpha, eigenvecs, r, ind = 0):
    v = 0.
    for i in range(NBASIS):
        v += eigenvecs[i, ind] * Gaussian(alpha[i], r)
    return v

# The basis functions are not orthogonal, so we have to solve the generalized eigenvalue problem, using the overlap:

H = np.zeros((NBASIS, NBASIS))
S = np.zeros((NBASIS, NBASIS))
for i in range(NBASIS):
    H[i, i] = Kinetic(alpha, i, i) + Coulomb(alpha, i, i)
    S[i, i] = Overlap(alpha, i, i)
    for j in range(i+1,NBASIS):
        H[i, j] = Kinetic(alpha, i, j) + Coulomb(alpha, i, j)
        S[i, j] = Overlap(alpha, i, j)
        H[j, i] = H[i, j]
        S[j, i] = S[i, j]

plt.imshow( S, origin='lower' )
plt.colorbar()
plt.savefig("OVERLAP.jpg", dpi=300)
plt.clf()

eigvals, eigvecs = eigh(H, S, eigvals_only=False)

print("HF Energies: E_0 = %1.3f a.u. (EXACT = 0.500)" % (eigvals[0]) )

# Plot Wavefunction
NTPS = 1000
r = np.linspace(0, 5, NTPS)
w = np.zeros((NTPS, 2))
for i in range(NTPS):
    w[i, 0] = np.abs(Wavefunction(alpha, eigvecs, r[i]))
    w[i, 1] = 1. / np.sqrt(np.pi) * np.exp(-r[i])

plt.plot(r, w),
plt.savefig("WAVEFUNCTION_H.jpg", dpi=300)
plt.clf()





###########################################

# ### Chap 4, Helium computation

###########################################

alpha = (0.298073, 1.242567, 5.782948, 38.47497)
NBASIS = len(alpha)

def TwoElectronSingleCenter(alpha, p, r, q, s):
    return 2. * m.pow(m.pi, 5. / 2.) / ((alpha[p]+alpha[q])*(alpha[r]+alpha[s])*m.sqrt(alpha[p]+alpha[q]+alpha[r]+alpha[s]))

H = np.zeros((NBASIS, NBASIS))
Ovr = np.zeros((NBASIS, NBASIS))
Q = np.zeros((NBASIS, NBASIS, NBASIS, NBASIS))

for i in range(NBASIS):
    for j in range(NBASIS):
        H[i, j] = Kinetic(alpha, i, j) + 2. * Coulomb(alpha, i, j) #the 2 is due of Z=2 for Helium
        Ovr[i, j] = Overlap(alpha, i, j)
        for k in range(NBASIS):
            for n in range(NBASIS):
                Q[i, j, k, n]=TwoElectronSingleCenter(alpha, i, j, k, n)

v = 1. / m.sqrt(Ovr.sum())
C = np.array([v, v, v, v]) # a choice for C to start with. Check the commented one instead
#C = np.array([1, 1, 1, 1])

F = np.zeros((NBASIS, NBASIS))
oldE = 100

for cycle in range(100):
    
    #for i in range(NBASIS):
    #    for j in range(NBASIS):
    #        F[i, j] = H[i, j]
    #        for k in range(NBASIS):
    #            for l in range(NBASIS):
    #                F[i, j] += Q[i, k, j, l] * C[k] * C[l]
                    
    F = H + np.einsum('ikjl,k,l', Q, C, C)                    
                    
    eigvals, eigvecs = eigh(F, Ovr, eigvals_only=False)

    C = eigvecs[:,0]

    #Eg = 0
    #for i in range(NBASIS):
    #    for j in range(NBASIS):
    #        Eg += 2 * C[i] * C[j] * H[i, j]
    #        for k in range(NBASIS):
    #            for l in range(NBASIS):
    #                Eg += Q[i, k, j, l] * C[i] * C[j] * C[k] * C[l]
                    
    Eg = 2 * np.einsum('ij,i,j', H, C, C) + np.einsum('ikjl,i,j,k,l', Q, C, C, C, C)
    
    if abs(oldE-Eg) < 1E-10:
        break
    
    oldE = Eg

print("HF Energies: E_0 = %1.3f a.u. (EXACT = ?????)" % (Eg) )



exit()


#####################################################

# HartreeFock with STO6G gets here -2.846 (but with a basis set with more Gaussians, you certainly can get better results), DFTAtom -2.8348 (again, that's normal for LDA: [NIST](https://www.nist.gov/pml/atomic-reference-data-electronic-structure-calculations/atomic-reference-data-electronic-7-1)) and the Variational Quantum Monte Carlo, -2.8759. The later 'beats' the result obtained here. 

# ### The first problem, H2+ (problem 4.8)
# 
# Now we have two centers so intergrals get more complex. You may simplify some integrals computation by using the already computed overlap, such optimizations are left out from here, they exist in the C++ project. For the H2+, as there is a single electron, we don't need a self-consistency loop.

def F0(t):
    if t==0:
        return 1.
    p = m.sqrt(t)
    a = 1. / p
    return a * m.sqrt(m.pi) / 2. * m.erf(p)

def Rp(alpha, beta, Ra, Rb):
    return (alpha*Ra + beta*Rb) / (alpha + beta)

def OverlapTwoCenters(alpha, beta, Ra, Rb):
    difR = Ra - Rb
    len2 = difR.dot(difR)
    aplusb = alpha + beta
    ab = alpha * beta / aplusb
    return m.pow(m.pi / aplusb, 3./2.) * m.exp(-ab * len2)

def KineticTwoCenters(alpha, beta, Ra, Rb):
    difR = Ra - Rb
    len2 = difR.dot(difR)
    aplusb = alpha + beta
    ab = alpha * beta / aplusb
    Ovr = m.pow(m.pi/aplusb, 3./2.) * m.exp(-ab * len2) # it's actually the overlap, check the OverlapTwoCenters
    return ab * (3. - 2. * ab * len2) * Ovr #this can be optimized with already computed overlap, see above

def Nuclear(alpha, beta, Ra, Rb, Rc, Z = 1.):
    aplusb = alpha + beta
    ab = alpha * beta / aplusb
    difR = Ra - Rb
    len2 = difR.dot(difR)
    difRc = Rp(alpha, beta, Ra, Rb) - Rc
    len2c = difRc.dot(difRc)
    K = m.exp(-ab*len2)
    return -2. * m.pi * Z / aplusb * K * F0(aplusb*len2c)

def TwoElectronTwoCenter(alpha, beta, gamma, delta, Ra, Rb, Rc, Rd):
    RP = Rp(alpha, gamma, Ra, Rc)
    RQ = Rp(beta, delta, Rb, Rd)
    alphaplusgamma = alpha + gamma
    betaplusdelta = beta + delta
    Rac = Ra - Rc
    Rbd = Rb - Rd
    Rpq = RP - RQ
    Racl2 = Rac.dot(Rac)
    Rbdl2 = Rbd.dot(Rbd)
    Rpql2 = Rpq.dot(Rpq)
    return 2. * m.pow(m.pi, 5./2.) / (alphaplusgamma * betaplusdelta * m.sqrt(alphaplusgamma+betaplusdelta)) * m.exp(-alpha*gamma/alphaplusgamma*Racl2 - beta*delta/betaplusdelta*Rbdl2) * F0(alphaplusgamma*betaplusdelta / (alphaplusgamma+betaplusdelta) * Rpql2)


alpha=(13.00773, 1.962079, 0.444529, 0.1219492)
NBASIS = len(alpha) # for each atom

H = np.zeros((NBASIS * 2, NBASIS * 2))
Ovr = np.zeros((NBASIS * 2, NBASIS * 2))
R0 = np.array([0, 0, 0])
R1 = np.array([1, 0, 0])
for i in range(NBASIS):
    a = alpha[i]    
    NBASISi = NBASIS + i
    for j in range(NBASIS):        
        b = alpha[j]        
        NBASISj = NBASIS + j
        Ovr[i, j] = OverlapTwoCenters(a, b, R0, R0)
        Ovr[NBASISi, j] = OverlapTwoCenters(a, b, R1, R0)
        Ovr[i, NBASISj] = OverlapTwoCenters(a, b, R0, R1)
        Ovr[NBASISi, NBASISj] = OverlapTwoCenters(a, b, R1, R1)
        H[i, j] = KineticTwoCenters(a, b, R0, R0) + Nuclear(a, b, R0, R0, R0) + Nuclear(a, b, R0, R0, R1)
        H[NBASISi, j] = KineticTwoCenters(a, b, R1, R0) + Nuclear(a, b, R1, R0, R0) + Nuclear(a, b, R1, R0, R1)
        H[i, NBASISj] = KineticTwoCenters(a, b, R0, R1) + Nuclear(a, b, R0, R1, R0) + Nuclear(a, b, R0, R1, R1)
        H[NBASISi, NBASISj] = KineticTwoCenters(a, b, R1, R1) + Nuclear(a, b, R1, R1, R0) + Nuclear(a, b, R1, R1, R1)


eigvals, eigvecs = eigh(H, Ovr, eigvals_only=False)

print(eigvals)


# ### Now the next problem, for H2 (problem 4.9) - here only the Hartree equation is solved
# 
# Filling the Q tensor looks ugly but I think it's better to leave it like that. If you want, you may make it prettier by observing that you can count from 0 to 0xF and use the proper bit to select to pass R0 or R1 and to select the proper sector of the tensor.

Q = np.zeros((NBASIS*2, NBASIS*2, NBASIS*2, NBASIS*2))
for i in range(NBASIS):
    a = alpha[i]
    NBASISi = NBASIS + i
    for j in range(NBASIS):
        b = alpha[j]
        NBASISj = NBASIS + j
        for k in range(NBASIS):
            c = alpha[k]
            NBASISk = NBASIS + k
            for n in range(NBASIS):
                NBASISl = NBASIS + n
                d = alpha[n]                
                Q[i, j, k, n]=TwoElectronTwoCenter(a, b, c, d, R0, R0, R0, R0)
                Q[i, j, k, NBASISl]=TwoElectronTwoCenter(a, b, c, d, R0, R0, R0, R1)
                Q[i, j, NBASISk, n]=TwoElectronTwoCenter(a, b, c, d, R0, R0, R1, R0)
                Q[i, j, NBASISk, NBASISl]=TwoElectronTwoCenter(a, b, c, d, R0, R0, R1, R1)
                Q[i, NBASISj, k, n]=TwoElectronTwoCenter(a, b, c, d, R0, R1, R0, R0)
                Q[i, NBASISj, k, NBASISl]=TwoElectronTwoCenter(a, b, c, d, R0, R1, R0, R1)
                Q[i, NBASISj, NBASISk, n]=TwoElectronTwoCenter(a, b, c, d, R0, R1, R1, R0)
                Q[i, NBASISj, NBASISk, NBASISl]=TwoElectronTwoCenter(a, b, c, d, R0, R1, R1, R1)
                Q[NBASISi, j, k, n]=TwoElectronTwoCenter(a, b, c, d, R1, R0, R0, R0)
                Q[NBASISi, j, k, NBASISl]=TwoElectronTwoCenter(a, b, c, d, R1, R0, R0, R1)
                Q[NBASISi, j, NBASISk, n]=TwoElectronTwoCenter(a, b, c, d, R1, R0, R1, R0)
                Q[NBASISi, j, NBASISk, NBASISl]=TwoElectronTwoCenter(a, b, c, d, R1, R0, R1, R1)
                Q[NBASISi, NBASISj, k, n]=TwoElectronTwoCenter(a, b, c, d, R1, R1, R0, R0)
                Q[NBASISi, NBASISj, k, NBASISl]=TwoElectronTwoCenter(a, b, c, d, R1, R1, R0, R1)
                Q[NBASISi, NBASISj, NBASISk, n]=TwoElectronTwoCenter(a, b, c, d, R1, R1, R1, R0)
                Q[NBASISi, NBASISj, NBASISk, NBASISl]=TwoElectronTwoCenter(a, b, c, d, R1, R1, R1, R1)              

v = 1. / m.sqrt(Ovr.sum())
C = np.array([v, v, v, v, v, v, v, v])

F = np.zeros((2*NBASIS, 2*NBASIS))
oldE = 100

for cycle in range(100):

    #for i in range(2*NBASIS):
    #    for j in range(2*NBASIS):
    #        F[i, j] = H[i, j]
    #        for k in range(2*NBASIS):
    #            for l in range(2*NBASIS):
    #                F[i, j] += Q[i, k, j, l] * C[k] * C[l]
    
    F = H + np.einsum('ikjl,k,l', Q, C, C) 

    eigvals, eigvecs = eigh(F, Ovr, eigvals_only=False)

    C = eigvecs[:,0]

    #Eg = 0
    #for i in range(2*NBASIS):
    #    for j in range(2*NBASIS):
    #        Eg += 2 * C[i] * C[j] * H[i, j]
    #        for k in range(2*NBASIS):
    #            for l in range(2*NBASIS):
    #                Eg += Q[i, k, j, l] * C[i] * C[j] * C[k] * C[l]

    Eg = C.dot(H + F).dot(C)
     
    if abs(oldE-Eg) < 1E-10:
        break
    
    oldE = Eg

print(Eg + 1) # +1 for nuclear repulsion energy

print(C)



#########################################################################

# ### Now the next problem, for H2 (problem 4.12) - but now with Hartree-Fock
# 
# Here there are a lot of optimizations suggested in the book - for the electron-electron integrals, one can reduce roughly to 1/8 the number of computed integrals using symmetry. Instead of solving the general eigenvalue problem each selfconsistency step you can go with solving only the regular eigenvalue problem, for details please check the book and the C++ HartreeFock project. I didn't want to go with those here.

Qt = np.zeros((NBASIS*2, NBASIS*2, NBASIS*2, NBASIS*2))
for p in range(2*NBASIS):
    for q in range(2*NBASIS):
        for r in range(2*NBASIS):
            for s in range(2*NBASIS):
                Qt[p, q, r, s] = 2. * Q[p, q, r, s] - Q[p, r, s, q]

C = np.array([v, v, v, v, v, v, v, v]) #reinitialize C

# as above, but Hartree-Fock with Qt instead of Q
F = np.zeros((2*NBASIS, 2*NBASIS))
oldE = 100

for cycle in range(100):

    #for i in range(2*NBASIS):
    #    for j in range(2*NBASIS):
    #        F[i, j] = H[i, j]
    #        for k in range(2*NBASIS):
    #            for l in range(2*NBASIS):
    #                F[i, j] += Qt[i, k, j, l] * C[k] * C[l]
    
    F = H + np.einsum('ikjl,k,l', Q, C, C) 

    eigvals, eigvecs = eigh(F, Ovr, eigvals_only=False)

    C = eigvecs[:,0]
    
    Eg = C.dot(H + F).dot(C)
       
    if abs(oldE-Eg) < 1E-10:
        break
    
    oldE = Eg

print(Eg + 1) # +1 for nuclear repulsion energy
print(C)

# We obtain the same result as above, despite using Hartree-Fock, because we have only two electrons, one with spin up, one with down, so no spin-exchange.
# 
# With the C++ Hartree-Fock project, restricted method with STO6G, the result is -1.0758. Again, with a better basis set the result could be much better. With the Variational Quantum Monte Carlo method I got -1.092, again, better than the result above.