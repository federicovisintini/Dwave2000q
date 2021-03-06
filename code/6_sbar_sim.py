import numpy as np
import pandas as pd
import scipy.interpolate
import scipy.integrate as ode
import pickle
import concurrent.futures
from settings import DATA_DIR, ANNEALING_SCHEDULE_XLS
import time

# PARAMETERS
h0_list = np.linspace(0.9, 1, 3)
st_list = np.linspace(0.66, 0.69, 4)
kz_list = np.linspace(0, 0.035, 71)
kx_list = np.linspace(0, 3e-6, 61)

omega_c = 8 * np.pi
anneal_pause_lenght = 1000 * np.arange(0, 21, 2)
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
    mean_E = [np.trace(rho @ sigmaz).real for rho in rho_list]

    # save results on file
    with open(DATA_DIR / f'sbar_sim_xz_h{h0:.2f}_s{s_bar:.2f}_kz{kz:.4f}_kx{kx:.2e}.pkl', 'wb') as f:
        pickle.dump(mean_E, f)
    return


if __name__ == '__main__':
    tic = time.time()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for h0 in h0_list:
            for s_bar in st_list:
                for kz in kz_list:
                    for kx in kx_list:
                        executor.submit(simulate_anneal, [h0, s_bar, kz, kx])

    print(time.time() - tic)
    print('END!!')

