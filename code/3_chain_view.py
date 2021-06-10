import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import scipy.interpolate
from settings import DATA_DIR

# if you want to restore the previous experimental data
with open(DATA_DIR / 'experimental_results_B.pkl', 'rb') as f:
    mean_E_exp = pickle.load(f)

print('Loaded data from <-', f.name)

x = np.linspace(1, 100, num=10)
y = np.array(mean_E_exp)

plt.figure("therm", figsize=(10, 7))
plt.plot(x, y, 'o', label='1 qubit (h=1)')
plt.ylabel(r'$\langle E \rangle $', fontsize=17)
plt.yticks(fontsize=12.5)
plt.xlabel(r'$t$ $(\mu s)$', fontsize=15)
plt.xticks(fontsize=12.5)
plt.title(r"$\bar{s}_t = 0.8$", fontsize=17)

with open(DATA_DIR / 'experimental_results_B2.pkl', 'rb') as f:
    mean_E_exp2 = pickle.load(f)

plt.plot(x, np.array(mean_E_exp2), 'o', label='2 qubits (J=1)')

k2s = []
mean_E_sims = []
# simulations
files = [x for x in DATA_DIR.glob('chain_sim_k*.pkl') if x.is_file()]
for file in files:
    with open(file, 'rb') as fp:
        mean_E_sim = pickle.load(fp)
        k2s.append(float(str(file)[71:-4]))
        mean_E_sims.append(mean_E_sim)

k2s, mean_E_sims = (list(t) for t in zip(*sorted(zip(k2s, mean_E_sims))))

for k2, mean_E_sim in zip(k2s, mean_E_sims):
    if 0.04 < k2 < 0.12:
        plt.plot(x, np.real(mean_E_sim), '--', label=f'k2={k2}')

plt.yscale('log')
plt.legend()

# LIVELLI ENERGETICI
sigmax = np.array([[0, 1], [1, 0]])
sigmaz = np.diag([1, -1])

df = pd.read_excel('../09-1216A-A_DW_2000Q_6_annealing_schedule.xls', sheet_name=1)
A = scipy.interpolate.CubicSpline(df['s'], df['A(s) (GHz)'])
B = scipy.interpolate.CubicSpline(df['s'], df['B(s) (GHz)'])
len_chain = 4


def H(s):
    ham = 0
    for i in range(len_chain):
        ham += A(s) * np.kron(np.kron(np.eye(2 ** i), sigmax), np.eye(2 ** (len_chain - i - 1))) / 2
    for i in range(len_chain - 1):
        ham += B(s) * np.kron(np.kron(np.kron(np.eye(2 ** i), sigmaz), sigmaz), np.eye(2 ** (len_chain - i - 2))) / 2

    return ham


"""
plt.figure(0.9)
eigenvalues, eigenvectors = np.linalg.eigh(H(0.9))
plt.scatter(range(len(eigenvalues)), eigenvalues)

plt.figure("spectrum")
xs = np.linspace(0, 1)
eigs = []
for x in xs:
    eigenvalues, eigenvectors = np.linalg.eigh(H(x))
    eigs.append(eigenvalues)

eigs = np.array(eigs)

for i in range(len(eigs[0])):
    plt.plot(xs, eigs[:, i], color='C0')
"""


# annealing schedule
time_f = 100
s_bar = 0.8


def t_to_s(t):
    half_time = time_f / 2
    if t < half_time:
        return 1 - t / half_time * (1 - s_bar)
    return 2 * s_bar - 1 + t / half_time * (1 - s_bar)


t_anneal = np.linspace(0, time_f, 101)
s_anneal = np.empty_like(t_anneal)
for i, t_ in enumerate(t_anneal):
    s_anneal[i] = t_to_s(t_)

plt.figure("annealing schedule")
plt.plot(t_anneal, s_anneal)
plt.xlabel("annealing time: t (µs)")
plt.ylabel("annealing parameter: s")
plt.title("annealing schedule")
plt.ylim((0, 1.1))

plt.show()
