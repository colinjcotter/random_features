from firedrake import *
import numpy as np
import matplotlib.pyplot as plt


class Parareal:
    """
    Parallel-in-time algorithm
    """

    def __init__(self, coarse_solver, fine_solver, V, u0, nG, K):
        """
        V: function space
        u0: initial condition
        nG: number of coarse timesteps
        K: number of parareal iterations
        """

        self.coarse_solver = coarse_solver
        self.fine_solver = fine_solver        
        self.nG = nG
        self.K = K

        # list to hold reference solution at coarse time points
        self.yref = [Function(V) for i in range(self.nG+1)]

        # list to hold coarse solution at coarse time points
        self.yG = [Function(V) for i in range(self.nG+1)]

        # list to hold coarse solution from previous iteration at
        # coarse time points
        self.yG_prev = [Function(V) for i in range(self.nG+1)]

        # list to hold solution at coarse time points
        self.soln = [Function(V) for i in range(self.nG+1)]

        # list to hold fine solution at coarse time points
        self.yF = [Function(V) for i in range(self.nG+1)]

        # Initialise everything
        self.yref[0].assign(u0)
        self.yG[0].assign(u0)
        self.yG_prev[0].assign(u0)
        self.yF[0].assign(u0)
        self.soln[0].assign(u0)

        # Functions for writing out to pvd files
        self.yG_out = Function(V, name="yG")
        self.yF_out = Function(V, name="yF")
        self.yref_out = Function(V, name="yref")

    def parareal(self):
        """
        Parareal calculation
        """

        # compute reference solution
        yref = self.yref
        for i in range(self.nG):
            # each application of the fine solver does nF timesteps
            yref[i+1].assign(self.fine_solver.apply(yref[i]))

        # get some things
        yG = self.yG
        yG_prev = self.yG_prev
        yF = self.yF
        soln = self.soln

        # set up output file and write out initial coarse solution and
        # initial reference solution (the same, as it's just the
        # initial condition!)
        outfile0 = File(f"output/burgers_parareal_K0.pvd")
        self.yG_out.assign(yG[0])
        self.yref_out.assign(yref[0])
        outfile0.write(self.yG_out, self.yref_out)

        # Initial coarse run through
        print(f"First coarse integrator iteration")
        for i in range(self.nG):
            yG[i+1].assign(self.coarse_solver.apply(yG[i]))
            soln[i+1].assign(yG[i+1])
            yG_prev[i+1].assign(yG[i+1])
            self.yG_out.assign(yG[i+1])
            self.yref_out.assign(yref[i+1])
            outfile0.write(self.yG_out, self.yref_out)

        for k in range(self.K):
            print(f"Iteration {k+1}")
            outfile = File(f"output/burgers_parareal_K{k+1}.pvd")
            self.yG_out.assign(soln[0])
            self.yref_out.assign(yref[0])
            outfile.write(self.yG_out, self.yref_out)
            # Predict and correct
            for i in range(self.nG):
                yF[i+1].assign(self.fine_solver.apply(soln[i]))
            for i in range(self.nG):
                yG[i+1].assign(self.coarse_solver.apply(soln[i]))
                soln[i+1].assign(yG[i+1] - yG_prev[i+1] + yF[i+1])
                print(errornorm(yG[i+1], yG_prev[i+1]))
                #print(errornorm(yG_correct[i+1], yref[i+1]))
            for i in range(self.nG):
                yG_prev[i+1].assign(yG[i+1])
                self.yG_out.assign(soln[i+1])
                self.yref_out.assign(yref[i+1])
                outfile.write(self.yG_out, self.yref_out)

class BurgersBE(object):
    """
    Solves Burgers equation using backwards Euler
    """
    def __init__(self, V, nu, dt, ndt):

        v = TestFunction(V)
        self.u = Function(V)
        self.u_ = Function(V)
        self.unp1 = Function(V)

        eqn = (self.u - self.u_) * v * dx + dt * (self.u_ * self.u_.dx(0) *  v * dx + nu * self.u.dx(0) * v.dx(0) * dx)

        prob = NonlinearVariationalProblem(eqn, self.u)
        self.solver = NonlinearVariationalSolver(prob)

        self.ndt = ndt

    def apply(self, u):

        self.unp1.assign(u)
        for n in range(self.ndt):
            self.u_.assign(self.unp1)
            self.solver.solve()
            self.unp1.assign(self.u)

        return self.unp1

class Burgers_rf(object):
    """
    Predicts a time T flow map for Burgers fitted from data
    """
    def __init__(self):
        from lib_features import random_f
        
        data = np.load("tdata.npy")
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
        self.u_out = None
        self.rf = rf
        
    def apply(self, u):
        print(u.dat.data[:].shape)
        out = self.rf.map(u.dat.data[:])
        if self.u_out:
            self.u_out.dat.data[:] = out
        else:
            self.u_out = u.copy()
            self.u_out.dat.data[:] = out
        return self.u_out

def gander_parareal():
    # settings to match Gander and Hairer paper
    n = 124
    mesh = PeriodicUnitIntervalMesh(n)

    # We choose degree 2 continuous Lagrange polynomials.
    V = FunctionSpace(mesh, "CG", 1)
    u0 = Function(V, name="Velocity")

    # Initial condition
    x = SpatialCoordinate(mesh)[0]
    u0.interpolate(sin(2*pi*x))

    # viscosity
    nu = 1/50.

    # end time
    tmax = 1

    # number of parareal iterations
    K = 10
    # number of coarse timesteps
    nG = 10
    # number of fine timesteps per coarse timestep
    nF = 10

    # coarse timestep
    dT = tmax / nG
    # fine timestep
    dt = dT / nF

    print("coarse timestep: ", dT)
    print("fine timestep: ", dt)

    #G = BurgersBE(V, nu, dT, 1)
    G = Burgers_rf()
    F = BurgersBE(V, nu, dt, nF)
    solver = Parareal(G, F, V, u0, nG, K)    
    solver.parareal()

if __name__ == "__main__":
    gander_parareal()
    #get_burgers_data()
