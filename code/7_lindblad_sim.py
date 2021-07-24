import numpy as np
import pandas as pd
import scipy.interpolate
import scipy.integrate as ode
import pickle
import concurrent.futures
from settings import DATA_DIR, ANNEALING_SCHEDULE_XLS, STS11, HS11
import time

# PARAMETERS
kz_list = np.linspace(0.3, 5, 20)
# kx_list = np.linspace(0, 3e-6, 21)

beta = 48 / 15
omega_c = 8 * np.pi
rho0 = np.array([1, 0, 0, 0], dtype=complex)

# definitions
df = pd.read_excel(ANNEALING_SCHEDULE_XLS, sheet_name=1)
A = scipy.interpolate.CubicSpline(df['s'], df['A(s) (GHz)'])
B = scipy.interpolate.CubicSpline(df['s'], df['B(s) (GHz)'])
sigmax = np.array([[0, 1], [1, 0]])
sigmaz = np.diag([1, -1])


def hamiltonian(t, s_bar, h0, pause_lenght):
    if t < 1000:
        s = 1 - (1 - s_bar) / 1000 * t
    elif 1000 <= t <= pause_lenght + 1000:
        s = s_bar
    else:
        s = s_bar + (1-s_bar) / 1000 * (t - pause_lenght)
    return A(s) * sigmax / 2 + h0 * B(s) * sigmaz / 2


def dissipator_term(L, p):
    L_dag = L.conj().T
    L2 = L_dag @ L
    return L @ p @ L_dag - p @ L2 / 2 - L2 @ p / 2


def liovillian(t, y, x, kx, s_bar, h0, anneal_pause):
    p = y.reshape(2, 2)

    # hamiltonian term
    ham = hamiltonian(t, s_bar, h0, anneal_pause)
    p_new = -1j * (ham @ p - p @ ham)

    # hamiltonian eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(ham)

    omega = eigenvalues[1] - eigenvalues[0]
    assert omega > 0

    g = eigenvectors[:, 0]
    e = eigenvectors[:, 1]

    # Lindblad operators
    Lz_ge = g.conj() @ sigmaz @ e * np.outer(g, e)
    # Lx_ge = g.conj() @ sigmax @ e * np.outer(g, e)

    # adding jump op contribution to p_new
    gamma = 2 * np.pi * omega * np.exp(- abs(omega) / omega_c)
    kz = 0.04 * omega * x
    p_new += gamma / (1 - np.exp(-beta * omega)) * kz * dissipator_term(Lz_ge, p)
    p_new -= gamma / (1 - np.exp(+beta * omega)) * kz * dissipator_term(Lz_ge, p)
    # p_new += gamma * kx * dissipator_term(Lx_ge, p)
    return p_new.reshape(4)


def simulate_anneal(args):
    h0 = args[0]
    s_bar = args[1]
    kz = args[2]
    kx = args[3]

    # performing numerical calculation: state evolution and obj write
    sigma_z_expectation_value = []
    for anneal_pause in range(11):
        solution = ode.solve_ivp(fun=liovillian, t_span=(0, 1000 * anneal_pause + 2000), y0=rho0,
                                 method='DOP853', rtol=1e-4, args=(kz, kx, s_bar, h0, 1000 * anneal_pause))

        rho = solution.y[:, -1].reshape(2, 2)
        sigma_z_expectation_value.append(np.trace(rho @ sigmaz).real)

    # save results on file
    with open(DATA_DIR / f'7_115kz/z_s{s_bar:.2f}_h{h0:.3f}_x{kz:.4f}.pkl', 'wb') as f:
        # with open(DATA_DIR / f'7_kzkx/zx_s{s_bar:.2f}_h{h0:.3f}_kz{kz:.4f}_kx{kx:.2e}.pkl', 'wb') as f:
        pickle.dump(sigma_z_expectation_value, f)
        print('save file ->', f.name)
    return


if __name__ == '__main__':
    tic = time.time()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for s_bar, hs in zip(STS11, HS11):
            for h0 in hs:
                for kz in kz_list:
                    # for kx in kx_list:
                    kx = 0
                    executor.submit(simulate_anneal, [h0, s_bar, kz, kx])

    print(f'{time.time() - tic:.1f} seconds')
    print('END!!')
