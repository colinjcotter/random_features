from numpy import *
from scipy import fft
from firedrake import ProgressBar

class random_f(object):
    def __init__(self, ng, nsamples, L, tau, alpha,
                 amplitude, beta, delta):
        self.tau = tau
        self.alpha = alpha
        self.s = arange(0.,ng)/ng
        self.ng = ng
        self.nsamples = nsamples
        self.L = L
        self.omega = fft.fftfreq(ng)*ng*2*pi/L
        self.ds = L/ng
        self.amplitude = amplitude
        self.beta = beta
        self.delta = delta
        self.E = amplitude*self.chi(abs(self.omega)*self.delta)

    def gen_theta(self):

        # spatial white noise
        theta = random.randn(self.ng)/self.ds**0.5
        # ifft
        thetab = fft.fft(theta)
        C = self.tau**(2*self.alpha-1)*(self.omega**2 + self.tau)**(-self.alpha)
        thetab *= C

        theta = real(fft.ifft(thetab))
        theta -= average(theta)
        return theta

    # energy modulation
    def chi(self, r):
        return minimum(r, (r+0.5)**-self.beta)

    def conv_theta(self, a, theta):
        af = fft.fft(a)
        conv = real(fft.ifft(af*self.E*self.thetaf, axis=1))
        #conv += average(a) - average(conv)
        return conv

    def build_thetas(self, nmodes):
        self.nmodes = nmodes
        thetas = []
        for i in ProgressBar("thetas").iter(range(nmodes)):
            thetas.append(self.gen_theta())
        self.thetas = array(thetas)
        self.thetaf = fft.fft(self.thetas, axis=1)
        
    def sigma(self, x):
        return where(x>0, x, exp(x) - 1)

    def build_A(self, llambda, data):
        self.llambda = llambda
        #A = zeros((self.nmodes, self.nmodes))
        #b = zeros(self.nmodes)
        A = zeros((self.ng, self.nsamples, self.nmodes))
        
        # set up the least squares
        for n in ProgressBar("sample").iter(range(self.nsamples)):
            a = data[:, n, 0]
            #for l in range(self.nmodes):
            #A[:, n, l] = self.conv_theta(a, self.thetas)
            A[:, n, :] = self.conv_theta(a, self.thetas).T
        A = self.sigma(A)
        self.A = A.reshape((self.nsamples*self.ng, self.nmodes))
        self.y = data[:, :, 1].reshape(self.nsamples*self.ng,)

    def solve_A(self):
        A = self.A
        y = self.y
        m, _ = A.shape
        c = (self.llambda/self.L*self.ng)**0.5
        B = vstack((A, c*eye(self.nmodes)))
        bp = concatenate((y, zeros(self.nmodes)))
        x = linalg.lstsq(B, bp, rcond=None)
        self.coeffs = x[0]

    def save(self, fname="rnd"):
        save(fname+"_thetas.npy", self.thetas)
        save(fname+"_coeffs.npy", self.coeffs)

    def load(self, fname=None):
        self.thetas = load(fname+"_thetas.npy")
        self.thetaf = fft.fft(self.thetas, axis=1)
        self.nmodes = self.thetas.shape[0]
        self.coeffs = load(fname+"_coeffs.npy")

    def map(self, a):
        assert(a.shape == (self.ng,))
        a_out = 0.*a
        c = self.coeffs
        F = self.sigma(self.conv_theta(a, self.thetas))
        print(shape(F))
        a_out = dot(c, F)
        print(linalg.norm(a-a_out))
        return a_out
