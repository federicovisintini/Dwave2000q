import numpy as np
import matplotlib.pyplot as plt
import pickle
import re
import pandas as pd
from scipy.interpolate import CubicSpline
from settings import DATA_DIR, ANNEALING_SCHEDULE_XLS


h = 0.3
df = pd.read_excel(ANNEALING_SCHEDULE_XLS, sheet_name=1)
A = CubicSpline(df['s'], df['A(s) (GHz)'])
B = CubicSpline(df['s'], df['B(s) (GHz)'])
sigmax = np.array([[0, 1], [1, 0]])
sigmaz = np.diag([1, -1])

# experiment
exp_files = [x for x in DATA_DIR.glob('6_temp_sbar_*_h*.pkl') if x.is_file()]
exp = {}
dexp = {}
for file in sorted(exp_files):
    m = re.search(r'6_temp_sbar_(.*)_h(.*).pkl', file.name)
    st = float(m.group(1))
    h = float(m.group(2))
    with open(file, 'rb') as fp:
        mean_spin, std_spin = pickle.load(fp)
    exp[st] = np.array(mean_spin) / h
    dexp[st] = np.array(std_spin) / h


def expected_sigma_z(s, beta, h_):
    ham = A(s) * sigmax / 2 + h_ * B(s) * sigmaz / 2
    eigenvalues, eigenvectors = np.linalg.eigh(ham)
    omega = eigenvalues[1] - eigenvalues[0]
    p = 1 / (1 + np.exp(beta * omega))
    ground = eigenvectors[:, 0]
    ecc = eigenvectors[:, 1]

    ground = np.array([0, 1])
    ecc = np.array([1, 0])
    rho = p * np.outer(ecc, ecc) + (1-p) * np.outer(ground, ground)
    return np.trace(rho @ sigmaz)


# annealing functions
x = np.linspace(0, 1, 1000)
plt.figure('annealing functions temp')
plt.plot(x, A(x), label='A(s) GHz')
plt.plot(x, h * B(x), label=f'{h} * B(s) GHz')
plt.plot([0, 1], [15 / 48] * 2, ls='--', label='E=kT')
# plt.plot([s_low, s_low], [0, 10], ls='--', c='black', alpha=0.5)
# plt.plot([s_high, s_high], [0, 10], ls='--', c='black', alpha=0.5)
plt.title('anneling functions')
plt.xlabel('s')
plt.ylabel('Energy (GHZ)')
plt.tight_layout()
plt.legend()

# figure
plt.figure("temperature", figsize=(15, 7.5))
plt.ylabel(r'$\langle \sigma_z \rangle $', fontsize=17)
plt.yticks(fontsize=12.5)
plt.xlabel(r'pause $\bar{s}$', fontsize=15)
plt.xticks(fontsize=12.5)
plt.title(r"1 qubit thermalization", fontsize=17)

anneal_pause_lenght = np.linspace(0, 20, 5)
s_pause = sorted([st for st in exp.keys()])

for i in range(5):
    y = [exp[st][i] for st in s_pause]
    dy = [dexp[st][i] for st in s_pause]
    time = anneal_pause_lenght[i]
    plt.plot(s_pause, y, marker='o', ls='-', alpha=time / 30 + 0.3, label=f't_pause={time} µs')

x = np.linspace(0, 1)
plt.plot(x, [expected_sigma_z(s, 48 / 16, 0.3) for s in x], ls='--', label='expected pop for T=16 mK')
plt.plot(x, [expected_sigma_z(s, 48 / 17, 0.3) for s in x], ls='--', label='expected pop for T=17 mK')
plt.plot(x, [expected_sigma_z(s, 48 / 18, 0.3) for s in x], ls='--', label='expected pop for T=18 mK')

plt.legend()
plt.tight_layout()


plt.figure('anneal schedule')
plt.plot([0, 1, 20, 21], [0, 0.4, 0.4, 1])
plt.title('anneal schedule of experiment')
plt.xlabel('time (µs)')
plt.ylabel('s')

plt.show()
