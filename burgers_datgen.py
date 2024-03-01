from firedrake import *
import matplotlib.pyplot as pp

ncells = 124
mesh = PeriodicIntervalMesh(ncells, 1)
V = FunctionSpace(mesh, "CG", 1)
VDG = FunctionSpace(mesh, "DG", 0)
pcg = PCG64(seed=538746341)
rg = RandomGenerator(pcg)

u = TrialFunction(V)
v = TestFunction(V)
tau = Constant(7.)
alpha = Constant(2)
d = Constant(1.0)
rhs = Function(VDG)
mfield = Function(V)
c0 = Constant(10.)

eqn1 = (inner(grad(u), grad(v)) + tau**2*u*v)*dx
L1 = v*tau**(2*alpha - d)*rhs*dx
m_prob1 = LinearVariationalProblem(eqn1, L1, mfield)
m_solver1 = LinearVariationalSolver(m_prob1)
eqn2 = (inner(grad(u), grad(v)) + tau**2*u*v)*dx
L2 = v*mfield*dx
m_prob2 = LinearVariationalProblem(lhs(eqn2), L2, mfield)
m_solver2 = LinearVariationalSolver(m_prob2)

u0 = Function(V)
u1 = Function(V)


def matern():
    xi = rg.normal(VDG)
    xi /= Constant((1./ncells)**0.5)
    rhs.assign(xi)
    m_solver1.solve()
    m_solver2.solve()

T = 1.0 #  simulation time
nu = Constant(1.0e-2)
dt = 1./ncells
timestep = Constant(dt)
F = (inner((u1 - u0)/timestep, v)
     + inner(u1*u1.dx(0), v) + nu*inner(grad(u1), grad(v)))*dx
b_prob = NonlinearVariationalProblem(F, u1)
b_solver = NonlinearVariationalSolver(b_prob)

def forward(u_in, u_out):
    u0.assign(u_in)

    t = 0.
    while t < T - 0.5*dt:
        t += dt
        b_solver.solve()
        u0.assign(u1)
    u_out.assign(u0)

u_out = Function(V)

nu = u_out.dat.data[:].size
nsamples = 512
import numpy as np
data = np.zeros((nu, nsamples, 2))

file0 = File("matern.pvd")
for i in ProgressBar("sample").iter(range(nsamples)):
    matern()
    mfield -= assemble(mfield*dx)
    mfield *= c0
    forward(mfield, u_out)
    file0.write(mfield, u_out)
    data[:, i, 0] = mfield.dat.data[:]
    data[:, i, 1] = u_out.dat.data[:]

X = Function(V)
x, = SpatialCoordinate(mesh)
X.interpolate(x)
print(X.dat.data)

np.save("xdat.npy", X.dat.data)
