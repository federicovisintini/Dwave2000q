import numpy as np
import pandas as pd
import scipy.interpolate
import scipy.integrate as ode
import time
import pickle
from tqdm import tqdm
from settings import DATA_DIR, ANNEALING_SCHEDULE_XLS

# PARAMETERS
kz_list = [0.0325, 0.033]
anneal_param_min_list = [0.69]
omega_c = 8 * np.pi
h0 = 0.95
benchmark = False

# definitions
sigmax = np.array([[0, 1], [1, 0]])
sigmaz = np.diag([1, -1])

df = pd.read_excel(ANNEALING_SCHEDULE_XLS, sheet_name=1)
A = scipy.interpolate.CubicSpline(df['s'], df['A(s) (GHz)'])
B = scipy.interpolate.CubicSpline(df['s'], df['B(s) (GHz)'])


def t_to_s(t):
    s1 = (1 - s_bar) * (1000 - t) / 1000 * np.heaviside(1000 - t, 0)
    s2 = s_bar
    s3 = (t - pause_time - 1000) / 1000 * np.heaviside(t - pause_time - 1000, 0)
    s = s1 + s2 + s3

    return s


def dissipator_term(L, p):
    L_dag = L.conj().T
    L2 = L_dag @ L
    return L @ p @ L_dag - p @ L2 / 2 - L2 @ p / 2


def liovillian(t, y):
    p = y.reshape(2, 2)
    s = t_to_s(t)
    ham = A(s) * sigmax / 2 + h0 * B(s) * sigmaz / 2

    # hamiltonian term
    p_new = -1j * (ham @ p - p @ ham)

    # hamiltonian eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(ham)

    omega = eigenvalues[1] - eigenvalues[0]
    assert omega > 0

    g = eigenvectors[:, 0]
    e = eigenvectors[:, 1]

    # Lindblad operators
    Lz_ge = g.conj() @ sigmaz @ e * np.outer(g, e)

    # adding jump op contribution to p_new
    gamma = 2 * np.pi * omega * np.exp(- abs(omega) / omega_c)
    p_new += gamma * kz * dissipator_term(Lz_ge, p)

    return p_new.reshape(4)


if benchmark:
    s_bar = 0.69  # [0.67, 0.68, 0.69]
    time_f = 10 * 1000
    kz = 0.02
    alpha = 1

    tic = time.time()
    rho0 = np.array([1, 0, 0, 0], dtype=complex)
    pause_time = time_f - 2000
    solution = ode.solve_ivp(fun=liovillian, t_span=(0, time_f), y0=rho0, method='RK45')
    final_rho = solution.y[:, -1].reshape(2, 2)
    toc = time.time()

    print(solution.message)
    print(f"Elapsed time: {toc-tic:.2f} seconds")
    print("Final time of integration:", solution.t[-1])
    print(final_rho.real)
    print('Mean energy:', np.trace(final_rho @ sigmaz).real)
else:
    for s_bar in tqdm(anneal_param_min_list, desc='s*', ncols=80):
        for kz in tqdm(kz_list, desc='kz', leave=False, ncols=80):
            # for alpha in tqdm(alpha_list, desc='alpha', leave=False, ncols=80):
            anneal_pause_lenght = 1000 * np.arange(0, 21, 2)
            mean_E = []

            for pause_time in anneal_pause_lenght:
                rho0 = np.array([1, 0, 0, 0], dtype=complex)

                # performing numerical calculation: state evolution and obj write
                time_f = pause_time + 2000
                solution = ode.solve_ivp(fun=liovillian, t_span=(0, time_f), y0=rho0, method='RK23')
                rho = solution.y[:, -1].reshape(2, 2)
                mean_E.append(np.trace(rho @ sigmaz).real)

            # save results on file
            with open(DATA_DIR / f'sbar_sim_h{h0}_s{round(s_bar, 4)}_kz{round(kz, 6)}.pkl', 'wb') as f:
                pickle.dump(mean_E, f)
