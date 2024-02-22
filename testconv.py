from numpy import *
from scipy import fft

ng = 128
L = 1.
s = arange(0.,ng)/ng
ds = L/ng
omega = fft.fftfreq(ng)*ng*2*pi/L

def gen_theta():

    # spatial white noise
    theta = random.randn(128)/ds**0.5
    
    # ifft
    thetab = fft.fft(theta)

    tau = 7.0
    alpha = 1
    
    C = tau**(2*alpha-1)*(omega**2 + tau)**(-alpha)
    
    thetab *= C

    theta = real(fft.ifft(thetab))
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
thetas = []
for i in modes:
    thetas.append(gen_theta())

lambda = 1.0
nmodes = 10
A = zeros((nmodes, nmodes))
b = zeros(ndata)

# set up the least squares
for n in ndata:
    phi = zeros((nmodes, ng))
    for l in range(nmodes):
        phi[l, :] = sigma(conv(a, thetas[l]))
    for l in range(nmodes):
        b[l] += dot(y[n, :], phi[l, :])*L/ng
        for i in range(nmodes):
            phi_i = sigma(conv(a, thetas[i]))
            A[l, i] += dot(phi[i, :], phi[l, :])*L/ng

# solve the least squares problem

B = vstack((A, lambda**0.5*eye(nmodes)))
bp = concatenate((b, zeros(nmodes)))

x = linalg.lstsq(B, bp)
