import numpy as np
import pandas as pd
import scipy.interpolate
import scipy.integrate as ode
import pickle
import concurrent.futures
from settings import DATA_DIR, ANNEALING_SCHEDULE_XLS  # STS, HS_LIST
import time

# PARAMETERS
kz_list = np.linspace(0.01, 0.03, 21)
kx_list = np.linspace(0, 3e-6, 21)

omega_c = 8 * np.pi
anneal_pause_lenght = 1000 * np.arange(0, 11)
rho0 = np.array([1, 0, 0, 0], dtype=complex)

# definitions
df = pd.read_excel(ANNEALING_SCHEDULE_XLS, sheet_name=1)
A = scipy.interpolate.CubicSpline(df['s'], df['A(s) (GHz)'])
B = scipy.interpolate.CubicSpline(df['s'], df['B(s) (GHz)'])
sigmax = np.array([[0, 1], [1, 0]])
sigmaz = np.diag([1, -1])


def dissipator_term(L, p):
    L_dag = L.conj().T
    L2 = L_dag @ L
    return L @ p @ L_dag - p @ L2 / 2 - L2 @ p / 2


def liovillian(t, y, kz, kx, omega_c, ham):
    p = y.reshape(2, 2)

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
    Lx_ge = g.conj() @ sigmax @ e * np.outer(g, e)

    # adding jump op contribution to p_new
    gamma = 2 * np.pi * omega * np.exp(- abs(omega) / omega_c)
    p_new += gamma * kz * dissipator_term(Lz_ge, p)
    p_new += gamma * kx * dissipator_term(Lx_ge, p)
    return p_new.reshape(4)


def simulate_anneal(args):
    h0 = args[0]
    s_bar = args[1]
    kz = args[2]
    kx = args[3]

    # performing numerical calculation: state evolution and obj write
    ham = A(s_bar) * sigmax / 2 + h0 * B(s_bar) * sigmaz / 2
    solution = ode.solve_ivp(fun=liovillian, t_span=(0, anneal_pause_lenght[-1]),
                             t_eval=anneal_pause_lenght, y0=rho0, method='DOP853',
                             rtol=1e-4, args=(kz, kx, omega_c, ham))

    rho_list = [solution.y[:, i].reshape(2, 2) for i in range(len(solution.y[0]))]
    sigma_z_expectation_value = [np.trace(rho @ sigmaz).real for rho in rho_list]

    # save results on file
    # with open(DATA_DIR / f'7_kz/z_s{s_bar:.2f}_h{h0:.3f}_kz{kz:.4f}.pkl', 'wb') as f:
    with open(DATA_DIR / f'7_kzkx/zx_s{s_bar:.2f}_h{h0:.3f}_kz{kz:.4f}_kx{kx:.2e}.pkl', 'wb') as f:
        pickle.dump(sigma_z_expectation_value, f)
    return


if __name__ == '__main__':
    tic = time.time()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        # for s_bar, hs in zip(STS, HS_LIST):
        s_bar = 0.60  # [0.60, 0.65, 0.70, 0.75]
        hs = [0.076, 0.20, 0.31, 0.417, 0.63, 0.84],
        # hs = [0.088, 0.18, 0.273, 0.366, 0.55, 0.73, 0.92],
        # hs = [0.080, 0.16, 0.24, 0.32, 0.48, 0.64, 0.80],
        # hs = [0.071, 0.142, 0.212, 0.283, 0.425, 0.565, 0.71]
        for h0 in hs:
            for kz in kz_list:
                for kx in kx_list:
                    executor.submit(simulate_anneal, [h0, s_bar, kz, kx])

    print(f'{time.time() - tic:.1f} seconds')
    print('END!!')
