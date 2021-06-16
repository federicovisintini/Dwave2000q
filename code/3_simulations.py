import pandas as pd
import numpy as np
import scipy.interpolate
import scipy.integrate as ode
import time
from tqdm import tqdm
import json
from settings import DATA_DIR

# PARAMETERS

temp = 15.8  # [13.6, 14.0, 14.4, 14.8, 15.2, 15.6, 16.0, 16.4]
k2 = 0.12  # [0.004, 0.006, 0.012, 0.028, 0.06, 0.12]
omega_c = 8 * np.pi

benchmark = False
simulation_1_qubit = True
simulation_2_qubits = True
save_result = True

# definitions
beta = 47.9924341590788 / temp
sigmax = np.array([[0, 1], [1, 0]])
sigmaz = np.diag([1, -1])

df = pd.read_excel('../09-1216A-A_DW_2000Q_6_annealing_schedule.xls', sheet_name=1)
A = scipy.interpolate.CubicSpline(5000 * df['s'], df['A(s) (GHz)'])
B = scipy.interpolate.CubicSpline(5000 * df['s'], df['B(s) (GHz)'])


def gamma(omega):
    """ Lindblad rates """
    # works for omega positive and negative (satisfy detailed balance)
    return 2 * np.pi * k2 * omega * np.exp(- abs(omega) / omega_c) / (1 - np.exp(-beta * omega))


def dissipator_term(L, p):
    L_dag = L.conj().T
    L2 = L_dag @ L
    return L @ p @ L_dag - p @ L2 / 2 - L2 @ p / 2


def liovillian1(t, y):
    p = y.reshape(2, 2)

    ham = A(t) * sigmax / 2 + B(t) * h * sigmaz / 2

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


def liovillian2(t, y):
    p = y.reshape(4, 4)

    ham = A(t) * (np.kron(np.eye(2), sigmax) + np.kron(sigmax, np.eye(2))) / 2
    ham += B(t) * h * np.kron(sigmaz, np.eye(2)) / 2
    ham += B(t) * hB * np.kron(np.eye(2), sigmaz) / 2
    ham += B(t) * J * np.kron(sigmaz, sigmaz) / 2

    # hamiltonian term
    p_new = -1j * (ham @ p - p @ ham)

    # hamiltonian eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(ham)

    for i in range(4):
        for j in range(i+1, 4):
            omega = eigenvalues[j] - eigenvalues[i]
            assert omega > 0

            g = eigenvectors[:, i]
            e = eigenvectors[:, j]

            # adding jump operators contribution to p_new
            g_sigmaz_e = g.conj() @ np.kron(sigmaz, np.eye(2)) @ e
            p_new += gamma(omega) * dissipator_term(g_sigmaz_e * np.outer(g, e), p)
            p_new += gamma(- omega) * dissipator_term(g_sigmaz_e * np.outer(e, g), p)

            g_sigmaz_e = g.conj() @ np.kron(np.eye(2), sigmaz) @ e
            p_new += gamma(omega) * dissipator_term(g_sigmaz_e * np.outer(g, e), p)
            p_new += gamma(- omega) * dissipator_term(g_sigmaz_e * np.outer(e, g), p)

    return p_new.reshape(16)


if benchmark:
    h = 0.01
    hB = 0
    J = 1
    # 2 qubits simulation
    tic = time.time()
    rho0 = np.ones(16, dtype=complex) / 4
    solution = ode.solve_ivp(fun=liovillian2, t_span=(0, 5000), y0=rho0, method='RK45')
    final_rho = solution.y[:, -1].reshape(4, 4)
    toc = time.time()

    print(solution.message)
    print(f"Elapsed time: {toc-tic:.2f} seconds")
    print("Final time of integration:", solution.t[-1])
    print(final_rho.real)


# problem biases and couplings
names = ["J=0.2", "J=1", "hB=0.1"]
hB_values = [0, 0, 0.1]
J0_values = [0.2, 1, 1]
x = np.linspace(-1, 1, 30)
hA = (x ** 3 + x) / 5
data = []

# print intro
print('Performing simulation')
print(f'T: {temp}')
print(f'k2: {k2}\n')

# 1 qubit simulation
if simulation_1_qubit:
    print(f"Name: '1 qubit'")
    rho0 = np.ones(4, dtype=complex) / 2

    # performing numerical calculation: state evolution and obj write
    for h in tqdm(hA, desc="h", ncols=80):
        solution = ode.solve_ivp(fun=liovillian1, t_span=(0, 5000), y0=rho0, method='RK23')
        rho = solution.y[:, -1].reshape(2, 2)

        obj = {
            "T": temp,
            "k2": k2,
            "name": "1 qubit",
            "hA": h,
            "hB": 0,
            "J": 0,
            "rho_real": rho.real.tolist(),
            "rho_imag": rho.imag.tolist()
        }
        data.append(obj)


# 2 qubit simulation
if simulation_2_qubits:
    for name, hB, J in zip(names, hB_values, J0_values):
        print(f"Name: '{name}'")
        rho0 = np.ones(16, dtype=complex) / 4

        # performing numerical calculation: state evolution and obj write
        for h in tqdm(hA, desc="h", ncols=80):
            solution = ode.solve_ivp(fun=liovillian2, t_span=(0, 5000), y0=rho0, method='RK23')
            rho = solution.y[:, -1].reshape(4, 4)
            # rho = rho0.reshape(4, 4)

            obj = {
                "T": temp,
                "k2": k2,
                "name": name,
                "hA": h,
                "hB": hB,
                "J": J,
                "rho_real": rho.real.tolist(),
                "rho_imag": rho.imag.tolist()
            }
            data.append(obj)

# save results on file
if save_result:
    with open(DATA_DIR / f'multiple_experiments_simulation_T{temp}_k{k2}.json', 'w') as fp:
        json.dump(data, fp, indent=4)
