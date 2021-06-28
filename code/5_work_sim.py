import pandas as pd
import numpy as np
import scipy.interpolate
import scipy.integrate as ode
import time
import pickle
import matplotlib.pyplot as plt
from settings import *

# simulation parameters
kz = 0.006
omega_c = 8 * np.pi
beta = 48 / 13.1
rho0 = np.array([0, 0, 0, 1], dtype=complex)
h = 0.15

# annealing schedule parameters
ti = 1 * 1000
ta = TA * 1000
tb = ta + 1000
tc = TC * 1000
td = TD * 1000
tf = td + 1000

si = S_LOW
sa = si
sb = S_HIGH
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
plt.tight_layout()
plt.xlabel('time (µs)')
plt.ylabel(r'$\langle \sigma_z \rangle$')
plt.title('work exctration measures v sim')

plt.plot(solution.t / 1000, polarization, label=r'Lindblad simulation with $k_z=0.006, T=13.1 mK$')

gibbs_a = 2 / (1 + np.exp(beta * B(sa))) - 1
gibbs_b = 2 / (1 + np.exp(beta * B(sb))) - 1

plt.plot([0, tf / 1000], [gibbs_a, gibbs_a], ls='--', label=r'thermal equilibrium for T= 13.1 mK, s=' + str(S_LOW))
plt.plot([0, tf / 1000], [gibbs_b, gibbs_b], ls='--', label=r'thermal equilibrium for T= 13.1 mK, s =' + str(S_HIGH))

for t in [ti, ta, tb, tc, td]:
    plt.plot([t / 1000, t / 1000], [-1, -0.9], ls='--', c='black', alpha=0.5)
    print(f't={t/1000}µs; pol={p(t)}')

# experiment
QLp = 0.98372856442; dQLp = 0.0002698672048704348
QLm = -0.9993189612; dQLm = 5.887529182877954e-05
QHp = 0.99682018618; dQHp = 0.00011311457361074311
QHm = -0.9998726114; dQHm = 2.4310899029991185e-05
Ap = -0.92210191082; dAp = 0.000608729445379175
Am = -0.92562959333; dAm = 0.000621892731678587
Bp = 0.719529642332; dBp = 0.0010672755689550594
Bm = -0.98649681528; dBm = 0.0002516637885110414
Cp = -0.91298873101; dCp = 0.0005975171447289941
Cm = -0.93845173934; dCm = 0.0005489863530870556
Dp = -0.92771190592; dDp = 0.0005993388474148749
Dm = -0.93340029397; dDm = 0.0005692088941221727


def evolution(sigma_z, sigma_u, sigma_d):
    # return (2 * x + measure_lm - measure_lp) / (measure_lm + measure_lp)
    p_up = (1 + sigma_z) / 2
    p_down = (1 - sigma_z) / 2

    return p_up * sigma_u + p_down * sigma_d


sigmaz_a = Am
plt.errorbar(ta / 1000, sigmaz_a, dAm, fmt='o', c='red', label="sampling 'markov'")
sigmaz_b = evolution(sigmaz_a, Bp, Bm)
plt.errorbar(tb / 1000, sigmaz_b, dBm + dBp, fmt='o', c='red')
sigmaz_c = evolution(sigmaz_b, Cp, Cm)
plt.errorbar(tc / 1000, sigmaz_c, dCm + dCp, fmt='o', c='red')
sigmaz_d = evolution(sigmaz_c, Dp, Dm)
plt.errorbar(td / 1000, sigmaz_d, dDm + dDp, fmt='o', c='red')

# naive measure
files = [x for x in DATA_DIR.glob('work_extraction_naive0_tf*.pkl') if x.is_file()]
anneal_schedules = []
for file in sorted(files):
    tf = float(file.name[25: -4])
    with open(file, 'rb') as fp:
        anneal_schedule, mean_spin, std_spin = pickle.load(fp)
        anneal_schedules.append(anneal_schedule)
    plt.errorbar(tf - 1, mean_spin, std_spin + abs(1 + mean_spin) / 100, fmt='.', c='green')

plt.legend()

plt.figure('anneals')
p_wrong_ip = (1 - QLp) / 2
for i in anneal_schedules:
    plt.plot([x[0] for x in i], [x[1] for x in i])
plt.title('naive measures schedule')
plt.xlabel('time (µs)')
plt.ylabel('s')


# other figures
plt.title('work extraction process')
plt.xlabel('time (µs)')

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
plt.xlabel('omega')
plt.ylabel('S')

plt.show()
