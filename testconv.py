from numpy import *
from lib_features import random_f

data = load("bdata.npy")
# data shape ng x nsamples x 2
ng = data.shape[0]
nsamples = data.shape[1]
L = 1.
tau = 7.0
alpha = 1
amplitude = 1.0
beta = 4.0
delta = 0.0025

rf = random_f(ng=ng, nsamples=nsamples, L=L,
              tau=tau, alpha=alpha,
                 amplitude=amplitude, beta=beta, delta=delta)

# forming the least squares system
# A, B are data arrays with each column an input output pair

# generate the thetas
nmodes = 1000
rf.build_thetas(nmodes)

llambda = 1.0e-9

# In the paper
# n' is testing pairs 4000
# n is training pairs 512
# m is the number of modes 4000

A, b = rf.build_A(llambda, data)

# solve the least squares problem
x = rf.solve_A(A, b)
rf.save(fname="test")
