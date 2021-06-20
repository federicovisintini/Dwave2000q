import pandas as pd
import numpy as np
import scipy.interpolate
import scipy.integrate as ode
import time
import pickle
from settings import DATA_DIR, ANNEALING_SCHEDULE_XLS

# PARAMETERS
kxs = [6e-10, 2e-9, 6e-9, 1e-8, 2e-8, 3e-8, 6e-8]  # 8e-8, 1e-7, 1.3e-7, 1.6e-7, 2e-7]
kzs = [0.001, 0.002, 0.005, 0.007, 0.01, 0.012, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04]
sts = [0.74]  # [0.60, 0.62, 0.64, 0.66, 0.68, 0.7]
omega_c = 8 * np.pi
benchmark = False

# definitions
sigmax = np.array([[0, 1], [1, 0]])
sigmaz = np.diag([1, -1])
num_sims = len(kxs) * len(kzs) * len(sts)

df = pd.read_excel(ANNEALING_SCHEDULE_XLS, sheet_name=1)
A = scipy.interpolate.CubicSpline(df['s'], df['A(s) (GHz)'])
B = scipy.interpolate.CubicSpline(df['s'], df['B(s) (GHz)'])


def t_to_s(t):
    half_time = time_f / 2
    if t < half_time:
        return 1 - t / half_time * (1 - s_bar)
    return 2 * s_bar - 1 + t / half_time * (1 - s_bar)


def dissipator_term(L, p):
    L_dag = L.conj().T
    L2 = L_dag @ L
    return L @ p @ L_dag - p @ L2 / 2 - L2 @ p / 2


def liovillian(t, y):
    p = y.reshape(2, 2)
    s = t_to_s(t)
    ham = A(s) * sigmax / 2 + B(s) * sigmaz / 2

    # hamiltonian term
    p_new = -1j * (ham @ p - p @ ham)

    # hamiltonian eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(ham)

    omega = eigenvalues[1] - eigenvalues[0]
    assert omega > 0

    g = eigenvectors[:, 0]
    e = eigenvectors[:, 1]

    # Lindblad operators
    Lx_ge = g.conj() @ sigmax @ e * np.outer(g, e)
    Lz_ge = g.conj() @ sigmaz @ e * np.outer(g, e)

    # adding jump op contribution to p_new
    gamma = 2 * np.pi * omega * np.exp(- abs(omega) / omega_c)
    p_new += gamma * kx * dissipator_term(Lx_ge, p)
    p_new += gamma * kz * dissipator_term(Lz_ge, p)

    return p_new.reshape(4)


if benchmark:
    s_bar = 0.66
    time_f = 10 * 1000
    kx = 2e-8
    kz = 0.02

    tic = time.time()
    rho0 = np.array([1, 0, 0, 0], dtype=complex)
    solution = ode.solve_ivp(fun=liovillian, t_span=(0, time_f), y0=rho0, method='RK45')
    final_rho = solution.y[:, -1].reshape(2, 2)
    toc = time.time()

    print(solution.message)
    print(f"Elapsed time: {toc-tic:.2f} seconds")
    print("Final time of integration:", solution.t[-1])
    print(final_rho.real)
    print('Mean energy:', np.trace(final_rho @ sigmaz).real)
else:
    # time evaluation
    count = 0
    tic = time.time()
    for s_bar in sts:
        for kx in kxs:
            for kz in kzs:
                # print intro
                print(f'\n{100 * count / num_sims:.2f} % completed')
                print(f'{(time.time() - tic) / 60:.2f} min passed.')
                print(f'Performing simulation with st: {s_bar}, (kx, kz): {kx, kz}')

                anneal_lenght = 1000 * np.linspace(5, 50, num=10)
                mean_E = []

                for time_f in anneal_lenght:
                    # print(f"annealing time: {time_f / 1000} µs")

                    rho0 = np.array([1, 0, 0, 0], dtype=complex)

                    # performing numerical calculation: state evolution and obj write
                    solution = ode.solve_ivp(fun=liovillian, t_span=(0, time_f), y0=rho0, method='RK23')
                    rho = solution.y[:, -1].reshape(2, 2)
                    mean_E.append(np.trace(rho @ sigmaz).real)

                # save results on file
                with open(DATA_DIR / f'coupling_sim_st{s_bar}_kx{kx}_kz{kz}.pkl', 'wb') as f:
                    pickle.dump(mean_E, f)

                print('Saved data into ->', f.name)
                count += 1
