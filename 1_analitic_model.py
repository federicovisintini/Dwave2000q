import pandas as pd
import numpy as np
import pickle
import scipy

from tqdm import tqdm

df = pd.read_excel('09-1216A-A_DW_2000Q_6_annealing_schedule.xls', sheet_name=1)
nominal_temp_ghz = 13.5 / 47.9924341590788

biases_ex, spin_up_ex, dspin_up_ex = pickle.load(open("data/results.pickle", "rb"))


def hamiltonian_term(ham, p):
    return ham @ p - p @ ham


def dissipator_term(L, p):
    L_dag = L.conj().T
    L2 = L_dag @ L
    return L @ p @ L_dag - p @ L2 / 2 - L2 @ p / 2


def liovillian(ham, p):
    # hamiltonian eigenvals and eigenvects
    eigenvalues, eigenvectors = np.linalg.eigh(ham)
    omega = eigenvalues[1] - eigenvalues[0]
    assert omega >= 0

    # Lindblad rates
    gammap = 2 * np.pi * omega * np.exp(- omega / omega_c) / (1 - np.exp(-beta * omega)) * g2
    gammam = gammap * np.exp(-beta * omega)

    # Lindblad operators
    a = eigenvectors[:, 0]
    b = eigenvectors[:, 1]
    a_sigmaz_b = a.conj() @ np.diag([1, -1]) @ b
    Lab = a_sigmaz_b * np.outer(a, b)
    Lba = a_sigmaz_b * np.outer(b, a)

    # computing p_new
    p_new = -1j * hamiltonian_term(ham, p)
    p_new += gammap * dissipator_term(Lab, p)
    p_new += gammam * dissipator_term(Lba, p)

    return p_new


# evolution with lindblad_evolution master equation
def lindblad_evolution(t, h):
    '''
    Given the time evolution 't' (np.array) and the bias 'h' (float)
    This function evolves the hamiltonian ground state according to Lindblad master equation.
    Return the final density matrix after evolution.
    '''
    # coefficients for linear annealing with time
    t = 1000 * t  # time (ns)

    A = scipy.interpolate.CubicSpline(t[-1] * df['s'], df['A(s) (GHz)'])
    B = scipy.interpolate.CubicSpline(t[-1] * df['s'], df['B(s) (GHz)'])

    # hamiltonian
    H0 = - np.array([[0, 1], [1, 0]]) / 2  # initial hamiltonian
    H1 = - h * np.diag([1, -1]) / 2  # final hamiltonian

    def H(x):
        return A(x) * H0 + B(x) * H1

    # initial state of evolution
    rho = np.array([[1, 1], [1, 1]]) / 2

    # results
    sigmax = []
    sigmaz = []
    err = 0
    img = 0

    # perform evolution
    dt = t[1] - t[0]

    for x in t:
        # 4th order Runge-Kutta
        k1 = dt * liovillian(H(x), rho)
        k2 = dt * liovillian(H(x + dt / 2), rho + k1 / 2)
        k3 = dt * liovillian(H(x + dt / 2), rho + k2 / 2)
        k4 = dt * liovillian(H(x + dt), rho + k3)

        # update rho
        rho = rho + k1 / 6 + k2 / 3 + k3 / 3 + k4 / 6

        # save expectation value of operators
        sigmax.append(rho[0][1] + rho[1][0])
        sigmaz.append(rho[0][0] - rho[1][1])

        # check runge kutta errors (normalization and hermitianity)
        err += abs(1 - np.trace(rho)) ** 2
        img += np.imag(sigmax[-1]) ** 2 + np.imag(sigmaz[-1]) ** 2

    # converting expectation values to array
    sigmaz = np.array(sigmaz)
    sigmax = np.array(sigmax)

    # check consitency
    assert(err < 1e-10)
    assert(img < 1e-10)

    # print('Errori:', err, img)

    return np.real(sigmaz), np.real(sigmax)


if __name__ == '__main__':
    temp = float(input("Insert temperature (mK): "))

    # function to evolve rho
    omega_c = 8 * np.pi
    g2 = 0.4 / (2 * np.pi)

    t = np.linspace(0, 20, 10 ** 5 + 1)

    beta = 47.9924341590788 / temp  # 1 / GHz

    biases_q = np.linspace(-0.3, 0.3, 50)
    p_up = []

    # performing numerical calculation
    for h in tqdm(biases_q):
        sigmaz, sigmax = lindblad_evolution(t, h)
        p_up.append(sigmaz[-1] / 2 + 0.5)

    # save results on file
    pickle.dump((biases_q, p_up), open(f"data/qu_lindblad_various_h{temp}.pickle", "wb"))
