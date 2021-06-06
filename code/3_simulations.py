import pandas as pd
import numpy as np
import json
import scipy.interpolate
from tqdm import tqdm
from settings import DATA_DIR

sigmax = np.array([[0, 1], [1, 0]])
sigmaz = np.diag([1, -1])


def H0(num_qubits):
    """Initial hamiltonian"""

    # sanity check
    if num_qubits == 0:
        raise ValueError('ci sono 0 qubits')

    H_init = 0
    # sum sigma_x over all qubits
    for i in range(num_qubits):
        H_init += np.kron(np.kron(np.eye(2**i), sigmax),
                          np.eye(2 ** (num_qubits - i - 1)))

    return H_init / 2


def H1(num_qubits, h, J):
    """Final hamiltonian"""

    # sanity check
    if num_qubits == 0:
        raise ValueError('ci sono 0 qubits')

    H_final = 0
    # sum sigma_z over all qubits, weighted by h
    for i in range(num_qubits):
        # final hamiltonian
        H_final += h[i] * np.kron(np.kron(np.eye(2 ** i), sigmaz),
                                  np.eye(2 ** (num_qubits - i - 1)))

    # sum sigma_z sigma_z over all couples, weighted with J
    for couple in J:
        qubit1, qubit2 = sorted(couple)
        H_final += J[couple] * np.kron(np.kron(np.kron(np.kron(
            np.eye(2 ** qubit1), sigmaz), np.eye(2 ** (qubit2 - qubit1 - 1))),
            sigmaz), np.eye(2 ** (num_qubits - qubit2 - 1)))

    return H_final / 2


def gamma(omega):
    """ Lindblad rates """
    # works for omega positive and negative (satisfy detailed balance)
    return 2 * np.pi * g2 * omega * np.exp(- abs(omega) / omega_c) /\
           (1 - np.exp(-beta * omega))


def hamiltonian_term(ham, p):
    return ham @ p - p @ ham


def dissipator_term(L, p):
    L_dag = L.conj().T
    L2 = L_dag @ L
    return L @ p @ L_dag - p @ L2 / 2 - L2 @ p / 2


def liovillian(ham, p):
    """
    Computes the liovillian of the system dp/dt = Liov(p)

    Parameters
    ----------
    ham: ndarray
        2D numpy array (same dim row and columns) representing the hamiltonian
    p: ndarray
        2D numpy array (same dim row and columns) representing the density matrix

    Returns
    -------
    p_new: ndarray
        the density matrix after evolution s.t. p_new - p = Liov(p)
    """
    num_qubits = int(np.log2(len(p)))

    # new density matrix
    p_new = -1j * hamiltonian_term(ham, p)

    # hamiltonian eigenvals and eigenvects
    eigenvalues, eigenvectors = np.linalg.eigh(ham)

    # sum over omega (we are evolving just the ground state)
    for energy_level in range(1, len(eigenvalues)):
        omega = eigenvalues[energy_level] - eigenvalues[0]
        assert omega > 0

        for qub in range(num_qubits):
            sigmaz_composit = np.kron(np.kron(np.eye(2 ** qub), sigmaz),
                                 np.eye(2 ** (num_qubits - qub - 1)))
            g = eigenvectors[:, 0]
            e = eigenvectors[:, energy_level]

            # Lindblad operators
            g_sigmaz_e = g.conj() @ sigmaz_composit @ e
            Li_ge = g_sigmaz_e * np.outer(g, e)
            Li_eg = g_sigmaz_e * np.outer(e, g)

            # adding jump op contribution to p_new
            p_new += gamma(omega) * dissipator_term(Li_ge, p)
            p_new += gamma(- omega) * dissipator_term(Li_eg, p)

    return p_new


def liovillian1(*args):
    return 0


def lindblad_evolution(t, h: list, J):
    """
    Perform the evolution of the system ground state according to Lindblad ME

    Parameters
    ----------
    t: ndarray
        1D numpy array representing time evolution measured in ms
        eg. t = np.linspace(0, 20, 10**5)
    h: list
        list of biases of qubits
        eg. h = [1, 0.3] means the qubit A has bias 1, the qubit B has bias 0.3
    J: dict
        couplings between the qubits, in the form:
            - key: tuple of qubits
            - value: streght of the coupling
        eg. J = {(0, 2): 0.3, (1, 2): -0.1}:
            the coupling between qubits 0 and 2 has strenght 0.3
                                        1 and 2 has strenght -0.1

    Returns
    -------
    list of tuples (sigmax, sigmaz)
        expectation values along (sigmax, sigmaz) of the i-th qubit
        where i is the position of the tuple in the list
    """

    num_qubits = len(h)

    # coefficients for linear annealing with time
    t = 1000 * t  # time (ns)

    A = scipy.interpolate.CubicSpline(t[-1] * df['s'], df['A(s) (GHz)'])
    B = scipy.interpolate.CubicSpline(t[-1] * df['s'], df['B(s) (GHz)'])

    # hamiltonian
    def H(x):
        return A(x) * H0(num_qubits) + B(x) * H1(num_qubits, h, J)

    # initial state of evolution
    rho = np.ones((2 ** num_qubits, 2 ** num_qubits), dtype=complex) / 2 ** num_qubits

    # result array and errors init
    err = 0

    # perform evolution
    dt = t[1] - t[0]

    for i in tqdm(range(len(t)), desc="t", leave=False, ncols=80):
        x = t[i]

        # 4th order Runge-Kutta
        k1 = dt * liovillian(H(x), rho)
        k2 = dt * liovillian(H(x + dt / 2), rho + k1 / 2)
        k3 = dt * liovillian(H(x + dt / 2), rho + k2 / 2)
        k4 = dt * liovillian(H(x + dt), rho + k3)

        # update rho
        rho = rho + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6
        err += abs(1 - np.trace(rho)) ** 2

    # check consistency
    assert (err < 1e-10)

    return rho


if __name__ == '__main__':
    # load data
    df = pd.read_excel('../09-1216A-A_DW_2000Q_6_annealing_schedule.xls', sheet_name=1)

    # function to evolve rho
    temp = 14.0
    g2 = 0.03
    omega_c = 8 * np.pi

    names = ["J=0.2", "J=1", "hB=0.1"]
    hB_values = [0, 0, 0.1]
    J0_values = [0.2, 1, 1]

    beta = 47.9924341590788 / temp  # 1 / GHz
    t = np.linspace(0, 5, 5 * 10 ** 4 + 1)

    print('T:', temp, 'K\nk2:', g2)

    spin_up = []
    x = np.linspace(-1, 1, 1)
    biases = (x ** 3 + x) / 5

    data = []

    print(f"1 qubit")
    # performing numerical calculation
    for hA in tqdm(biases, desc="h", ncols=80):
        rho = lindblad_evolution(t, [hA], {})

        obj = {
            "T": temp,
            "k2": g2,
            "name": "1 qubit",
            "hA": hA,
            "hB": 0,
            "J": 0,
            "rho_real": rho.real.tolist(),
            "rho_imag": rho.imag.tolist()
        }
        data.append(obj)

    for name, hB, J in zip(names, hB_values, J0_values):
        print(name)
        # performing numerical calculation
        for hA in tqdm(biases, desc="h", ncols=80):
            rho = lindblad_evolution(t, [hA, hB], {(0, 1): J})

            obj = {
                "T": temp,
                "k2": g2,
                "name": name,
                "hA": hA,
                "hB": hB,
                "J": J,
                "rho_real": rho.real.tolist(),
                "rho_imag": rho.imag.tolist()
            }
            data.append(obj)

    # save results on file
    with open(DATA_DIR / f'multiple_experiments_simulation_T{temp}_k{g2}.json', 'w') as fp:
        json.dump(data, fp, indent=4)
