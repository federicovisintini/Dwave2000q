import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import re
from scipy.interpolate import CubicSpline
from settings import DATA_DIR, ANNEALING_SCHEDULE_XLS


def t_to_s(t, pause_time, min_s):
    return (1 - min_s) * (1-t) * np.heaviside(1-t, 0) + min_s + (1 - min_s) * (t-pause_time-1) * np.heaviside(t-pause_time-1, 0)


df = pd.read_excel(ANNEALING_SCHEDULE_XLS, sheet_name=1)
A = CubicSpline(df['s'], df['A(s) (GHz)'])
B = CubicSpline(df['s'], df['B(s) (GHz)'])

sigmax = np.array([[0, 1], [1, 0]])
sigmaz = np.diag([1, -1])

# experiment
exp_files = [x for x in DATA_DIR.glob('sbar_0.6*_h*.pkl') if x.is_file()]
exp = {}
dexp = {}
for file in sorted(exp_files):
    m = re.search(r'sbar_(0.\d+)_h([\.\d]*).pkl', file.name)
    st = float(m.group(1))
    h = float(m.group(2))
    with open(file, 'rb') as fp:
        mean_spin, std_spin = pickle.load(fp)
    exp[(h, st)] = np.array(mean_spin) / h
    dexp[(h, st)] = np.array(std_spin) / h + 0.01


# simulation
sim_files = [x for x in DATA_DIR.glob('sbar_sim_h*_s*_kz*.pkl') if x.is_file()]
sim = {}

for file in sorted(sim_files):
    m = re.search(r'sbar_sim_h([\.\d]*)_s(0.\d+)_kz(0.\d+).pkl', file.name)
    h = float(m.group(1))
    st = float(m.group(2))
    kz = float(m.group(3))
    with open(file, 'rb') as fp:
        mean_spin = pickle.load(fp)
    sim[(h, st, kz)] = np.array(mean_spin)


# chi 2
chi2 = {}
for h, s_bar, kz in sim.keys():
    chi2[(h, s_bar, kz)] = np.sum((exp[(h, s_bar)] - sim[(h, s_bar, kz)]) ** 2 / dexp[(h, s_bar)] ** 2)

simulated_experiments = set((h0, st) for (h0, st, kz) in sim.keys())

for h, st in simulated_experiments:
    kz_list = []
    chi2_list = []
    for (h0, st0, kz) in sim.keys():
        if h == h0 and st == st0:
            kz_list.append(kz)
            chi2_list.append((chi2[h0, st0, kz]))

    chi2_func = CubicSpline(kz_list, chi2_list)
    kz_sample = np.linspace(min(kz_list), max(kz_list), 300)
    chi2_sample = [chi2_func(k) for k in kz_sample]

    chi2_treshold = min(chi2_sample) + np.sqrt(2 * min(chi2_sample))
    k_opt = kz_sample[np.argmin(chi2_sample)]
    k_err = min(abs(chi2_func.solve(chi2_treshold) - k_opt))

    ham = A(st) * sigmax / 2 + h * B(st) * sigmaz / 2
    eigenvalues, eigenvectors = np.linalg.eigh(ham)
    omega = eigenvalues[1] - eigenvalues[0]

    plt.figure('chi2')
    plt.plot(kz_sample, chi2_sample, label=f'h={h}, st={st}')

    plt.figure('kz vs omega')
    plt.errorbar(omega, k_opt, k_err, marker='.', label=f'h={h}, st={st}')


plt.figure('chi2')
plt.xlabel(r'$k_z$')
plt.ylabel(r'$\chi^2$')
plt.ylabel(r'$\chi^2(\omega)$')
plt.legend()

plt.figure('kz vs omega')
plt.xlabel(r'$\omega$')
plt.ylabel('$k_z$')
plt.title(r'$k_z(\omega)$')
plt.legend()

# figure
plt.figure("therm", figsize=(15, 7.5))
plt.ylabel(r'$\langle E \rangle $', fontsize=17)
plt.yticks(fontsize=12.5)
plt.xlabel('pause time (µs)', fontsize=15)
plt.xticks(fontsize=12.5)
plt.title(r"1 qubit thermalization", fontsize=17)
x = list(range(0, 21, 2))

c_dict = {}

for i, (h, st) in enumerate(simulated_experiments):
    c_dict[(h, st)] = 'C' + str(i)
    plt.errorbar(x, exp[h, st], dexp[h, st], c=c_dict[(h, st)], marker='o', ls='', label=f'h={h}, st={st}')

for h, st, kz in sim.keys():
    plt.plot(x, sim[(h, st, kz)], c=c_dict[(h, st)], ls='--')


plt.legend()
plt.tight_layout()

x = np.linspace(0, 50)
plt.figure('anneal schedule')
plt.plot(x, t_to_s(x, 48, 0.7))
plt.title('anneal schedule of experiment')
plt.xlabel('time (µs)')
plt.ylabel('s')

plt.show()

"""
simulated_experiments = set((h0, st) for (h0, st, kz) in sim.keys())
omega_list = []
for h, s in simulated_experiments:
    ham = A(s) * sigmax / 2 + h * B(s) * sigmaz / 2
    eigenvalues, eigenvectors = np.linalg.eigh(ham)
    omega_list.append(eigenvalues[1] - eigenvalues[0])

k_mean1 = []
k_err1 = []
plt.figure('chi2')

    chi2 = CubicSpline(kzs1, chi2s1)
    x = np.linspace(min(kzs1), max(kzs1), 300)
    chi2_list = [chi2(k) for k in x]
    k_opt = x[np.argmin(chi2_list)]
    k_mean1.append(k_opt)

    chi2 = CubicSpline(kzs1, chi2s1)
    x = np.linspace(min(kzs1), max(kzs1), 300)
    chi2_list = [chi2(k) for k in x]
    k_opt = x[np.argmin(chi2_list)]
    k_mean2.append(k_opt)

    chi2_treshold = min(chi2_list) + 5
    k_dx = 0
    for i, chi in enumerate(chi2_list):
        if chi < chi2_treshold:
            k_dx = x[i]
    k_err.append(k_dx - k_opt)
    plt.plot(x, chi2_list)
plt.xlabel(r'$k_z$')
plt.ylabel(r'$\chi^2$')


plt.figure('kz vs omega')
# anneal_param_min_list
plt.errorbar(omega_list, k_mean, k_err, ls='', marker='.')
plt.xlabel(r'$\omega$')
plt.ylabel('$k_z$')
plt.title('kz vs omega')

# figure
plt.figure("therm", figsize=(15, 7.5))
plt.ylabel(r'$\langle E \rangle $', fontsize=17)
plt.yticks(fontsize=12.5)
plt.xlabel('pause time (µs)', fontsize=15)
plt.xticks(fontsize=12.5)
plt.title(r"1 qubit thermalization", fontsize=17)
x = list(range(0, 21, 2))

c_dict = {
    0.67: 'C0',
    0.68: 'C1',
    0.69: 'C2'
}

for i, (st, kz) in enumerate(sim1.keys()):
    plt.errorbar(x, exp1[st], dexp1[st], c=c_dict[st], marker='o', ls='', label=str(st))
    plt.plot(x, sim1[(st, kz)], c=c_dict[st], ls='--', label=f's={st}, k={kz}')


# plt.legend()
plt.tight_layout()

x = np.linspace(0, 50)
plt.figure('anneal schedule')
plt.plot(x, t_to_s(x, 48, 0.7))
plt.xlabel('time (µs)')
plt.ylabel('s')


plt.show()
"""