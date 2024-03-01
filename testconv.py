from numpy import *
from scipy import fft
from firedrake import ProgressBar

data = load("bdata.npy")
# data shape ng x nsamples x 2
ng = data.shape[0]
nsamples = data.shape[1]
L = 1.
s = arange(0.,ng)/ng
ds = L/ng
omega = fft.fftfreq(ng)*ng*2*pi/L

def gen_theta():

    # spatial white noise
    theta = random.randn(ng)/ds**0.5
    
    # ifft
    thetab = fft.fft(theta)

    tau = 7.0
    alpha = 1
    
    C = tau**(2*alpha-1)*(omega**2 + tau)**(-alpha)
    
    thetab *= C

    theta = real(fft.ifft(thetab))
    theta -= average(theta)
    return theta

# energy modulation
amplitude = 1.0
beta = 4.0
delta = 0.0025
def chi(r):
    return minimum(r, (r+0.5)**-beta)

E = amplitude*chi(abs(omega)*delta)

def conv_theta(a, theta):
    af = fft.fft(a)
    thetaf = fft.fft(theta)
    conv = real(fft.ifft(af*thetaf*E))
    conv += average(a) - average(conv)
    return conv
    
# forming the least squares system
# A, B are data arrays with each column an input output pair

# generate the thetas
nmodes = 1000
thetas = []
print(ng)
for i in ProgressBar("theta").iter(range(nmodes)):
    thetas.append(gen_theta())

llambda = 1.0e-6
A = zeros((nmodes, nmodes))
b = zeros(nmodes)

def sigma(x):
    return where(x>0, x, exp(x) - 1)

# in the paper
# n' is testing pairs 4000
# n is training pairs 512
# m is the number of modes 4000

# set up the least squares
for n in ProgressBar("sample").iter(range(nsamples)):
    a = data[:, n, 0]
    y = data[:, n, 1]
    phi = zeros((nmodes, ng))
    for l in range(nmodes):
        phi[l, :] = sigma(conv_theta(a, thetas[l]))
    for l in range(nmodes):
        b[l] += dot(y, phi[l, :])*L/ng
        for i in range(nmodes):
            phi_i = sigma(conv_theta(a, thetas[i]))
            A[l, i] += dot(phi[i, :], phi[l, :])*L/ng

# solve the least squares problem

B = vstack((A, llambda**0.5*eye(nmodes)))
bp = concatenate((b, zeros(nmodes)))

x = linalg.lstsq(B, bp, rcond=None)
