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
E = amplitude*(s**2+0.01)*exp(-(s/0.7)**2)

def conv_theta(a, theta):
    af = fft.fft(a)
    thetaf = fft.fft(theta)
    conv = real(fft.ifft(af*thetaf*E))
    return conv
    
import matplotlib.pyplot as pp
for i in range(10):
    theta = gen_theta()
    a = gen_theta()
    conv = conv_theta(a, theta)
    #pp.plot(a)
    pp.plot(conv)
    pp.show()
