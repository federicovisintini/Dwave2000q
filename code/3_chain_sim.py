import pandas as pd
import numpy as np
import scipy.interpolate
import scipy.integrate as ode
import time
import pickle
from settings import DATA_DIR

# PARAMETERS
k2 = 0.6
s_bar = 0.8  # [0.65, 0.7, 0.75, 0.8]
temp = 1
omega_c = 8 * np.pi
benchmark = False

# definitions
sigmax = np.array([[0, 1], [1, 0]])
sigmaz = np.diag([1, -1])

df = pd.read_excel('../09-1216A-A_DW_2000Q_6_annealing_schedule.xls', sheet_name=1)
A = scipy.interpolate.CubicSpline(df['s'], df['A(s) (GHz)'])
B = scipy.interpolate.CubicSpline(df['s'], df['B(s) (GHz)'])
h = 1


def t_to_s(t):
    half_time = time_f / 2
    if t < half_time:
        return 1 - t / half_time * (1 - s_bar)
    return 2 * s_bar - 1 + t / half_time * (1 - s_bar)


def gamma(omega):
    """ Lindblad rates """
    # works for omega positive and negative (satisfy detailed balance)
    if temp == 0:
        if omega > 0:
            return 2 * np.pi * k2 * omega * np.exp(- abs(omega) / omega_c)
        return 0
    beta = 47.9924341590788 / temp
    return 2 * np.pi * k2 * omega * np.exp(- abs(omega) / omega_c) / (1 - np.exp(-beta * omega))


def dissipator_term(L, p):
    L_dag = L.conj().T
    L2 = L_dag @ L
    return L @ p @ L_dag - p @ L2 / 2 - L2 @ p / 2


def liovillian1(t, y):
    p = y.reshape(2, 2)
    s = t_to_s(t)
    ham = A(s) * sigmax / 2 + B(s) * h * sigmaz / 2

    # hamiltonian term
    p_new = -1j * (ham @ p - p @ ham)

    # hamiltonian eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(ham)

    omega = eigenvalues[1] - eigenvalues[0]
    assert omega > 0

    g = eigenvectors[:, 0]
    e = eigenvectors[:, 1]

    # Lindblad operators
    g_sigmaz_e = g.conj() @ sigmaz @ e
    L_ge = g_sigmaz_e * np.outer(g, e)
    L_eg = g_sigmaz_e * np.outer(e, g)

    # adding jump op contribution to p_new
    p_new += gamma(omega) * dissipator_term(L_ge, p)
    p_new += gamma(- omega) * dissipator_term(L_eg, p)

    return p_new.reshape(4)


if benchmark:
    time_f = 1000
    tic = time.time()
    rho0 = np.array([1, 0, 0, 0], dtype=complex)
    solution = ode.solve_ivp(fun=liovillian1, t_span=(0, time_f), y0=rho0, method='RK45')
    final_rho = solution.y[:, -1].reshape(2, 2)
    toc = time.time()

    print(solution.message)
    print(f"Elapsed time: {toc-tic:.2f} seconds")
    print("Final time of integration:", solution.t[-1])
    print(final_rho.real)

else:
    # print intro
    print(f'Performing simulation w k2: {k2}\n')

    anneal_lenght = 1000 * np.linspace(1, 100, num=10)
    mean_E = []

    for time_f in anneal_lenght:
        print(f"annealing time: {time_f}")

        rho0 = np.array([1, 0, 0, 0], dtype=complex)

        # performing numerical calculation: state evolution and obj write
        solution = ode.solve_ivp(fun=liovillian1, t_span=(0, time_f), y0=rho0, method='RK23')
        rho = solution.y[:, -1].reshape(2, 2)
        mean_E.append(np.trace(rho @ sigmaz))

    # save results on file
    with open(DATA_DIR / f'chain_sim_k{k2}_st{s_bar}_T{temp}.pkl', 'wb') as f:
        pickle.dump(mean_E, f)

    print('Saved data into ->', f.name)
