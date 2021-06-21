import pandas as pd
import numpy as np
import scipy.interpolate
import scipy.integrate as ode
import time
import matplotlib.pyplot as plt
from settings import ANNEALING_SCHEDULE_XLS

# simulation parameters
kz = 0.02
omega_c = 8 * np.pi
beta = 48 / 13.5
rho0 = np.array([0, 0, 0, 1], dtype=complex)
h = 0.15

# annealing schedule parameters
ti = 1 * 1000
ta = 30 * 1000
tb = ta + 1000
tc = 110 * 1000
td = 190 * 1000
tf = td + 1000

si = 0.68
sa = si
sb = 0.72
sc = sb
sd = sa


# DEFINITIONS
sigmax = np.array([[0, 1], [1, 0]])
sigmaz = np.diag([1, -1])

df = pd.read_excel(ANNEALING_SCHEDULE_XLS, sheet_name=1)
A = scipy.interpolate.CubicSpline(df['s'], df['A(s) (GHz)'])
B = scipy.interpolate.CubicSpline(df['s'], df['B(s) (GHz)'] * h)


def integrand(x):
    return - kz ** 2 * x / (1 - np.exp(- beta * x)) * np.exp(-abs(x) / omega_c)  # / (10 - x)


def S(omega):
    # ans0, err, *_ = ode.quad(lambda x: integrand(x) / (x - omega), -np.inf, -20)
    ans1, err, *_ = ode.quad(integrand, -100, 1000, weight='cauchy', wvar=omega)
    return ans1


S0 = S(0)


def annealing_schedule(t):
    """Return s given t"""
    if t < ti:
        return 1 - t / ti * (1 - si)
    elif ti < t <= ta:
        return si + (sa - si) * (t - ti) / (ta - ti)
    elif ta < t <= tb:
        return sa + (sb - sa) * (t - ta) / (tb - ta)
    elif tb < t <= tc:
        return sb + (sc - sb) * (t - tb) / (tc - tb)
    elif tc < t <= td:
        return sc + (sd - sc) * (t - tc) / (td - tc)
    elif td < t <= tf:
        return sd + (1 - sd) * (t - td) / (tf - td)


def dissipator_term(L, p):
    L_dag = L.conj().T
    L2 = L_dag @ L
    return L @ p @ L_dag - p @ L2 / 2 - L2 @ p / 2


def liovillian(t, y):
    p = y.reshape(2, 2)
    s = annealing_schedule(t)
    ham = A(s) * sigmax / 2 + B(s) * sigmaz / 2

    # hamiltonian eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(ham)

    omega = eigenvalues[1] - eigenvalues[0]
    assert omega > 0
    assert abs(omega) < 15

    g = eigenvectors[:, 0]
    e = eigenvectors[:, 1]

    # Lindblad operators
    Lz_ge = g.conj() @ sigmaz @ e * np.outer(g, e)
    Lz_eg = g.conj() @ sigmaz @ e * np.outer(e, g)
    # L_ee = e.conj() @ sigmaz @ e * np.outer(e, e)
    # L_gg = g.conj() @ sigmaz @ g * np.outer(g, g)

    # hamiltonian term + lamb shift hamiltonian
    gamma = 2 * np.pi * omega * np.exp(- abs(omega) / omega_c)
    # tot_ham = ham(Lz_ge.conj().T @ Lz_ge * S(gamma) + Lz_eg.conj().T @ Lz_eg * S(-gamma))
    # tot_ham += (L_ee.conj().T + L_gg.conj().T) @ (L_ee + L_gg) * S0
    p_new = -1j * (ham @ p - p @ ham)

    # adding jump op contribution to p_new
    p_new += gamma * kz * dissipator_term(Lz_ge, p) / (1 - np.exp(-beta * omega))
    p_new -= gamma * kz * dissipator_term(Lz_eg, p) / (1 - np.exp(beta * omega))

    return p_new.reshape(4)


tic = time.time()
solution = ode.solve_ivp(fun=liovillian, t_span=(0, tf), y0=rho0, method='RK23', atol=1e-8, rtol=1e-5)
rhos = [solution.y[:, i].reshape(2, 2) for i in range(len(solution.t))]
polarization = np.array([np.trace(rho @ sigmaz).real for rho in rhos])
hamiltonian_z = np.array([B(annealing_schedule(t)) for t in solution.t])
energy = hamiltonian_z * polarization
p = scipy.interpolate.CubicSpline(solution.t, polarization)
toc = time.time()

print(solution.message)
print(f"Elapsed time: {toc-tic:.2f} seconds")

xx = np.linspace(0, tf)
yy = [annealing_schedule(tt) for tt in xx]

plt.figure('annealing schedule')
plt.plot(xx / 1000, yy)
plt.title('anneling schedule')
plt.xlabel('time (µs)')
plt.ylabel('s')

plt.figure('work extraction process')
plt.plot(solution.t / 1000, polarization, label=r'$<\sigma_z>$')
# plt.plot(solution.t / 1000, energy, label='<E>')
# plt.plot(solution.t / 1000, - hamiltonian_z, label='-B(t)')

gibbs_a = 2 / (1 + np.exp(beta * B(sa))) - 1
gibbs_b = 2 / (1 + np.exp(beta * B(sb))) - 1

plt.plot([0, tf / 1000], [gibbs_a, gibbs_a], ls='--')
plt.plot([0, tf / 1000], [gibbs_b, gibbs_b], ls='--')

for t in [ti, ta, tb, tc, td]:
    plt.plot([t / 1000, t / 1000], [-1, -0.9], ls='--', c='black', alpha=0.5)
    print(f't={t/1000}µs; pol={p(t)}')

plt.title('work extraction process')
plt.xlabel('time (µs)')
plt.legend()

xx = np.linspace(-30, 300, 10000)
plt.figure('integrand of S(omega)')
plt.plot(xx, integrand(xx))

xx = np.linspace(-15, 15, 1000)
yy = []
yy_precise = []

for i in xx:
    a1 = S(i)
    yy.append(a1)


plt.figure('S(omega)')
plt.plot(xx, yy)

plt.show()
