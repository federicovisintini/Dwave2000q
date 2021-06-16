import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import scipy.interpolate
import re
from settings import DATA_DIR, ANNEALING_SCHEDULE_XLS


s_bar = 0.75

# 1 qubit
st1 = []
mean_E_sim1 = []
files = [x for x in DATA_DIR.glob('experimental_results_B1_st*_num102050.pkl') if x.is_file()]
for file in sorted(files):
    st1.append(float(file.name[26: -14]))
    with open(file, 'rb') as fp:
        mean_E_sim1.append(pickle.load(fp))

# 2 qubits
st2 = []
mean_E_sim2 = []
files = [x for x in DATA_DIR.glob('experimental_results_B2_st*_num50850.pkl') if x.is_file()]
for file in sorted(files):
    st2.append(float(file.name[26: -13]))
    with open(file, 'rb') as fp:
        mean_E_sim2.append(pickle.load(fp))

# figure
plt.figure("therm", figsize=(10, 7))
plt.ylabel(r'$\langle E \rangle $', fontsize=17)
plt.yticks(fontsize=12.5)
plt.xlabel(r'$t (\mu s)$', fontsize=15)
plt.xticks(fontsize=12.5)
plt.title(r"1 qubit thermalization", fontsize=17)

x = np.linspace(1, 100, num=10)
for st, meanE in zip(st1, mean_E_sim1):
    plt.plot(x, np.array(meanE), 'o', label=r'1 qubit (h=1), $\bar{s}=$' + str(st))

for st, meanE in zip(st2, mean_E_sim2):
    pass
    # plt.plot(x, np.array(meanE), 'o', label=r'2 qubits (J=1), $\bar{s}=$' + str(st))


# SIMULATIONS
T_sim = []
k2_sim = []
st_sim = []
mean_E_sims = []

files = [x for x in DATA_DIR.glob('chain_sim_k*_st*_T*.pkl') if x.is_file()]
for file in files:
    with open(file, 'rb') as fp:
        mean_E_sim = pickle.load(fp)
    m = re.search(r'chain_sim_k(0\.\d+)_st(0.\d+)_T(\d+).pkl', file.name)
    k2_sim.append(m.group(1))
    st_sim.append(m.group(2))
    T_sim.append(m.group(3))
    mean_E_sims.append(mean_E_sim)

for idx in np.argsort(st_sim):
    plt.plot(x, np.real(mean_E_sims[idx]), '--', label=f'k2={k2_sim[idx]}, S={st_sim[idx]}')

# plt.yscale('log')
plt.legend()

# BEST FIT as func of S_T
st_fit = [0.65, 0.7, 0.75, 0.8]
k2_fit = [0.023, 0.033, 0.043, 0.068]
dk2_fit = [0.002] * 4

plt.figure('fit')
plt.errorbar(st_fit, k2_fit, dk2_fit, fmt='o')
plt.title('Spin-bath coupling for reverse anneal')
plt.xlabel(r'$\bar{s}$')
plt.ylabel('$k_{fit}^2$')

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


# annealing functions
df = pd.read_excel(ANNEALING_SCHEDULE_XLS, sheet_name=1)
nominal_temp_ghz = 13.5 / 47.9924341590788

# plot annealing functions vs annealing parameter and plotting nominal temperature for comparison
plt.figure("anneling functions", figsize=(8, 6))
plt.plot(df['s'], df['A(s) (GHz)'], label='A(s)')
plt.plot(df['s'], df['B(s) (GHz)'], label='B(s)')
plt.plot(s_bar * np.ones(50), np.linspace(0, 12), linestyle='--', color='black')
plt.plot(np.linspace(0, 1), nominal_temp_ghz * np.ones(50), linestyle='--', label='E = $k_B$T')
plt.title('Annealing functions')
plt.xlabel('s')
plt.ylabel('Energy (GHz)')
plt.legend()


# annealing schedule
time_f = 100


def t_to_s(t):
    half_time = time_f / 2
    if t < half_time:
        return 1 - t / half_time * (1 - s_bar)
    return 2 * s_bar - 1 + t / half_time * (1 - s_bar)


t_anneal = np.linspace(0, time_f, 100)
s_anneal = np.empty_like(t_anneal)
for i, t_ in enumerate(t_anneal):
    s_anneal[i] = t_to_s(t_)

plt.figure("annealing schedule")
plt.plot(t_anneal, s_anneal)
plt.xlabel("annealing time: t (µs)")
plt.ylabel("annealing parameter: s")
plt.title(r"annealing schedule $\bar{s}=$" + str(s_bar))
plt.ylim((0, 1.1))

plt.show()
