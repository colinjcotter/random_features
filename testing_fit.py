from numpy import *
from lib_features import random_f
import matplotlib.pyplot as pp

data = load("tdata.npy")
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

rf.load(fname="test")

for i in range(20):
    a = data[:, i, 0]
    a_T = data[:, i, 1]
    a_out = rf.map(a)
    pp.plot(a, 'r-')
    pp.plot(a_T, 'k-.')
    pp.plot(a_out, 'b--')
    pp.legend(["in", "out", "out_rf"])
    pp.savefig("plots_"+str(i)+".png")
    pp.close()
