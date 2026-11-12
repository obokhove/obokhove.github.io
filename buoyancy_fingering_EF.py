#
# MATH5453M FEM Exercise 3 Part II, step 4 (first scheme):
# Semi-implicit Euler-forward CG1-FEM for the buoyancy-fingering system, eqs (3a)-(3b)
#
#   g_t = ( e g / sqrt(e+g) )_zz
#   e_t = ( e e_z / sqrt(e+g) )_z - e g / sqrt(e+g) + eps (1-e) sqrt(e+g)
#
# g(0,t) = g(H,t) = g0 (inhomogeneous Dirichlet), e_z = 0 at both ends (natural BC).
# Adapted from Poissons_eq_v2.py (mesh -> space -> weak form -> assemble/solve -> output).
#
# Time-step criterion (maximum-principle idea, as in the finite-difference exercise):
#   g-equation: g_t = F_zz with F = e g/sqrt(e+g) and F_g = e(2e+g) / (2 (e+g)^{3/2}) >= 0
#   e-equation: diffusivity D_e = e / sqrt(e+g)
#   explicit diffusion is monotone (max principle) if  dt <= h^2 / (2 max(F_g, D_e)).
# With CG1 + MASS LUMPING (lump=True) the FEM scheme is algebraically the finite-difference scheme,
# so this criterion carries over exactly. With the CONSISTENT mass matrix (lump=False) there is no
# discrete maximum principle and explicit stability needs dt <= h^2 / (6 max(F_g, D_e)) (factor 3 stricter).
#
from firedrake import *
import numpy as np
import time as timer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- parameters (from the sheet)
H = 2000.0
Tend = 100000.0
nz = 4000                    # h = 0.5
degree = 1
r = 50.0
eps = 1.0/r
m = 10
a = 0.001
g0 = 0.0218
gp = a*g0                    # g'
ep = gp                      # e'
e0 = 0.5*(1 - g0*(1 + r) + np.sqrt((1 - g0*(1 + r))**2 + 4*g0))   # (1f)

lump = True                  # True: lumped mass (= FD scheme, max principle); False: consistent mass + CG solves
cfl = 0.9                    # safety factor on the maximum-principle bound (<= 1)
n_out = 400                  # number of stored output times (space-time plot)
vtk_every = 0                # >0: write .pvd every vtk_every outputs

h = H/nz
print(f'H={H}, nz={nz}, h={h}, e0={e0:.5f}, eps={eps}, lump={lump}')

# ---------------------------------------------------------------- mesh, spaces, fields
mesh = IntervalMesh(nz, H)   # boundary ids: 1 -> z=0, 2 -> z=H
V = FunctionSpace(mesh, 'CG', degree)
z, = SpatialCoordinate(mesh)
w = TestFunction(V)          # w_g
v = TestFunction(V)          # v_e

g_n = Function(V, name='g')
e_n = Function(V, name='e')
g_new = Function(V, name='g_new')
e_new = Function(V, name='e_new')
Fh = Function(V, name='F')   # CG1 interpolant of the flux e g/sqrt(e+g)

k = 2*pi*m/H
g_n.interpolate(g0 - k*gp*cos(k*z))     # (1c)
e_n.interpolate(e0 - k*ep*cos(k*z))

bcs_g = [DirichletBC(V, Constant(g0), 1), DirichletBC(V, Constant(g0), 2)]   # g = g0 at z=0,H
for bc in bcs_g:
    bc.apply(g_n)            # make the (tiny) boundary mismatch of (1c) consistent with (1d)

S_n = sqrt(e_n + g_n)

# ---------------------------------------------------------------- weak forms
# (3a): w (g^{n+1}-g^n)/dt + w_z (e^n g^n / S^n)_z = 0           -> fully explicit in g
R_g = inner(grad(w), grad(Fh))*dx                                # Fh = interpolant of e^n g^n / S^n
# (3b): v (e^{n+1}-e^n)/dt + v_z e^n e^n_z/S^n + v e^{n+1} g^n/S^n + eps v (e^{n+1}-1) S^n = 0
R_e = inner(grad(v), e_n*grad(e_n)/S_n)*dx                        # explicit diffusion of e

if lump:
    ml = assemble(w*dx).dat.data_ro.copy()    # lumped mass = row sums of the consistent mass matrix
else:
    dtc = Constant(1.0)
    gtr, etr = TrialFunction(V), TrialFunction(V)
    a_g = w*gtr/dtc*dx
    L_g = w*g_n/dtc*dx - R_g
    a_e = (v*etr/dtc + v*etr*g_n/S_n + eps*v*etr*S_n)*dx
    L_e = (v*e_n/dtc + eps*v*S_n)*dx - R_e
    params = {'ksp_type': 'cg', 'pc_type': 'jacobi', 'ksp_rtol': 1e-12}
    solver_g = LinearVariationalSolver(LinearVariationalProblem(a_g, L_g, g_new, bcs=bcs_g), solver_parameters=params)
    solver_e = LinearVariationalSolver(LinearVariationalProblem(a_e, L_e, e_new), solver_parameters=params)

def dt_max():
    """Maximum-principle time-step bound from the current nodal values."""
    g = g_n.dat.data_ro; e = e_n.dat.data_ro
    S = np.sqrt(e + g)
    Fg = e*(2*e + g)/(2*S**3)
    De = e/S
    D = max(Fg.max(), De.max())
    return cfl*h**2/((2.0 if lump else 6.0)*D)

def step(dt):
    if lump:
        Fh.interpolate(e_n*g_n/S_n)
        rg = assemble(R_g).dat.data_ro
        re = assemble(R_e).dat.data_ro
        g = g_n.dat.data_ro; e = e_n.dat.data_ro
        S = np.sqrt(e + g)
        g_new.dat.data[:] = g - dt*rg/ml                                   # (3a), lumped
        for bc in bcs_g:
            bc.apply(g_new)
        e_new.dat.data[:] = (e/dt + eps*S - re/ml)/(1.0/dt + g/S + eps*S)  # (3b), lumped; e^{n+1} in the 2 sinks
    else:
        dtc.assign(dt)
        Fh.interpolate(e_n*g_n/S_n)
        solver_g.solve()
        solver_e.solve()
    g_n.assign(g_new)
    e_n.assign(e_new)

# ---------------------------------------------------------------- time loop
xs = Function(V).interpolate(z).dat.data_ro.copy()
order = np.argsort(xs)
xs = xs[order]
out_times = np.linspace(0.0, Tend, n_out + 1)
G = np.empty((n_out + 1, len(xs))); E = np.empty_like(G)
G[0] = g_n.dat.data_ro[order]; E[0] = e_n.dat.data_ro[order]
outfile = VTKFile('buoyancy_EF.pvd') if vtk_every > 0 else None

t = 0.0; nstep = 0; dt_min = np.inf; t0 = timer.time()
for i in range(1, n_out + 1):
    while t < out_times[i] - 1e-9*Tend:
        dt = min(dt_max(), out_times[i] - t)
        step(dt)
        t += dt; nstep += 1; dt_min = min(dt_min, dt)
    G[i] = g_n.dat.data_ro[order]; E[i] = e_n.dat.data_ro[order]
    if not np.isfinite(G[i]).all() or not np.isfinite(E[i]).all():
        raise RuntimeError(f'Blow-up at t={t:.1f}: reduce cfl.')
    if outfile is not None and i % vtk_every == 0:
        outfile.write(g_n, e_n, time=t)
    if i % 10 == 0:
        print(f't={t:9.1f}  steps={nstep}  dt={dt:.3f}  g in [{G[i].min():.5f},{G[i].max():.5f}]  '
              f'e in [{E[i].min():.5f},{E[i].max():.5f}]  wall={timer.time()-t0:.0f}s', flush=True)

print(f'done: {nstep} steps, smallest dt={dt_min:.4f}, wall time {timer.time()-t0:.1f}s')
np.savez('buoyancy_EF.npz', z=xs, t=out_times, g=G, e=E)

# ---------------------------------------------------------------- figures
sel = np.linspace(0, n_out, 6, dtype=int)
fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
for i in sel:
    ax[0].plot(xs, G[i], label=f't={out_times[i]:.0f}')
    ax[1].plot(xs, E[i], label=f't={out_times[i]:.0f}')
ax[0].set_ylabel('g(z,t)'); ax[1].set_ylabel('e(z,t)'); ax[1].set_xlabel('z')
ax[0].legend(ncol=3); fig.tight_layout(); fig.savefig('buoyancy_EF_profiles.png', dpi=150)

fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for A, F_, nm in zip(ax, (G, E), ('g', 'e')):
    im = A.pcolormesh(xs[::8], out_times, F_[:, ::8], shading='auto')
    A.set_xlabel('z'); A.set_title(nm); fig.colorbar(im, ax=A)
ax[0].set_ylabel('t'); fig.tight_layout(); fig.savefig('buoyancy_EF_spacetime.png', dpi=150)
print('saved buoyancy_EF_profiles.png, buoyancy_EF_spacetime.png, buoyancy_EF.npz')
