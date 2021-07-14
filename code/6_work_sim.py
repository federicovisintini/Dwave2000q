import pandas as pd
import numpy as np
import scipy.interpolate
import scipy.integrate as ode
import time
import pickle
import matplotlib.pyplot as plt
from settings import *

# simulation parameters
kz = 0.008
omega_c = 8 * np.pi
T = 13.5
beta = 48 / T
rho0 = np.array([0, 0, 0, 1], dtype=complex)

s_low = 0.61
s_high = 0.66
h = 0.4

ta = (20 + 1) * 1000
tb = ta + 1000
tc = tb + 50 * 1000

# DEFINITIONS
sigmax = np.array([[0, 1], [1, 0]])
sigmaz = np.diag([1, -1])

df = pd.read_excel(ANNEALING_SCHEDULE_XLS, sheet_name=1)
A = scipy.interpolate.CubicSpline(df['s'], df['A(s) (GHz)'])
B = scipy.interpolate.CubicSpline(df['s'], df['B(s) (GHz)'] * h)


def annealing_schedule(t):
    """Return s given t"""
    if t < 1000:
        return 1 - t / 1000 * (1 - s_low)
    elif 1000 < t <= ta:
        return s_low
    elif ta < t <= tb:
        return s_low + (s_high - s_low) * (t - ta) / (tb - ta)
    elif tb < t <= tc:
        return s_high
    elif tc < t <= tc + 1000:
        return s_high + (1 - s_high) * (t - tc) / 1000


def dissipator_term(L, p):
    L_dag = L.conj().T
    L2 = L_dag @ L
    return L @ p @ L_dag - p @ L2 / 2 - L2 @ p / 2


def liovillian(t, y):
    p = y.reshape(2, 2)
    s = annealing_schedule(t)
    ham = A(s) * sigmax / 2 + h * B(s) * sigmaz / 2

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

    # hamiltonian term + lamb shift hamiltonian
    gamma = 2 * np.pi * omega * np.exp(- abs(omega) / omega_c)
    p_new = -1j * (ham @ p - p @ ham)

    # adding jump op contribution to p_new
    p_new += gamma * kz * dissipator_term(Lz_ge, p) / (1 - np.exp(-beta * omega))
    p_new -= gamma * kz * dissipator_term(Lz_eg, p) / (1 - np.exp(beta * omega))

    return p_new.reshape(4)


tic = time.time()
solution = ode.solve_ivp(fun=liovillian, t_span=(0, tc + 1000), y0=rho0, method='DOP853', rtol=1e-6, atol=1e-8)
rhos = [solution.y[:, i].reshape(2, 2) for i in range(len(solution.t))]
polarization = np.array([np.trace(rho @ sigmaz).real for rho in rhos])
p = scipy.interpolate.CubicSpline(solution.t, polarization)
toc = time.time()

print(solution.message)
print(f"Elapsed time: {toc-tic:.2f} seconds")

xx = np.linspace(0, tc + 1000)
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

plt.plot(solution.t / 1000, polarization, label=r'Lindblad simulation with $k_z=0.008, T=13.5 mK$')

gibbs_a = 2 / (1 + np.exp(beta * h * B(s_low))) - 1
gibbs_b = 2 / (1 + np.exp(beta * h * B(s_high))) - 1

ham = A(s_low) * sigmax / 2 + h * B(s_low) * sigmaz / 2
eigenvalues, eigenvectors = np.linalg.eigh(ham)
omega_a = eigenvalues[1] - eigenvalues[0]
gibbs_aa = 2 / (1 + np.exp(beta * omega_a)) - 1

ham = A(s_high) * sigmax / 2 + h * B(s_high) * sigmaz / 2
eigenvalues, eigenvectors = np.linalg.eigh(ham)
omega_b = eigenvalues[1] - eigenvalues[0]
gibbs_bb = 2 / (1 + np.exp(beta * omega_b)) - 1

plt.plot([0, tc / 1000 + 1], [gibbs_a, gibbs_a], ls='--', label=f'thermal equilibrium for T={T} mK, s={s_low}, along z')
plt.plot([0, tc / 1000 + 1], [gibbs_b, gibbs_b], ls='--', label=f'thermal equilibrium for T={T} mK, s={s_high}, along z')
plt.plot([0, tc / 1000 + 1], [gibbs_aa, gibbs_aa], ls='--', label=f'thermal equilibrium for T={T} mK, s={s_low}, along xz')
plt.plot([0, tc / 1000 + 1], [gibbs_bb, gibbs_bb], ls='--', label=f'thermal equilibrium for T={T} mK, s={s_high}, along xz')


for t in [ta, tb, tc]:
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
Cm = -0.93845173934; dCm = 0.000548986353087055


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

# naive measure
files = [x for x in DATA_DIR.glob('6_work_extraction_naive0_tf*.pkl') if x.is_file()]
anneal_schedules = []
for file in sorted(files):
    tf = float(file.name[25: -4])
    with open(file, 'rb') as fp:
        anneal_schedule, mean_spin, std_spin = pickle.load(fp)
        anneal_schedules.append(anneal_schedule)
    plt.errorbar(tf - 1, mean_spin, std_spin + abs(1 + mean_spin) / 100, fmt='.', c='green')

plt.legend()

QLp = 0.90970601; dQLp = 0.0006727
QHp = 0.98822146; dQHp = 0.0002236
Ap = -0.99536501; dAp = 0.00015535
Am = -0.99643312; dAm = 0.00012796
Bp = 0.071337579; dBp = 0.00157151
Bm = -0.99649191; dBm = 0.00013584
Cp = -0.99446349; dCp = 0.00015948
Cm = -0.99417442; dCm = 0.00017088

energy_high = B(s_high)
energy_low = B(s_low)

work_pp = h * (energy_high - energy_low) / 2
work_mm = - h * (energy_high - energy_low) / 2
work_pm = h * (energy_high + energy_low) / 2
work_mp = - h * (energy_high + energy_low) / 2

p_pp = (1 + Am) / 2 * (1 + Bp) / 2
p_mm = (1 - Am) / 2 * (1 - Bm) / 2
p_pm = (1 + Am) / 2 * (1 - Bp) / 2
p_mp = (1 - Am) / 2 * (1 + Bm) / 2

plt.figure('work extraction')
plt.plot([work_pp, work_pp], [0, p_pp], label='pp')
plt.plot([work_mm, work_mm], [0, p_mm], label='mm')
plt.plot([work_pm, work_pm], [0, p_pm], label='pm')
plt.plot([work_mp, work_mp], [0, p_mp], label='mp')

plt.legend()

plt.show()
