import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import json
import re
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from scipy.interpolate import CubicSpline
from settings import DATA_DIR, ANNEALING_SCHEDULE_XLS


def t_to_s(t, pause_time, min_s):
    return (1 - min_s) * (1-t) * np.heaviside(1-t, 0) + min_s + (1 - min_s) *\
           (t-pause_time-1) * np.heaviside(t-pause_time-1, 0)


def gap(st, h):
    hamiltonian = A(st) * sigmax / 2 + h * B(st) * sigmaz / 2
    eigenvals, eigenvects = np.linalg.eigh(hamiltonian)
    energy_gap = eigenvals[1] - eigenvals[0]
    return energy_gap


gray = '0.4'
df = pd.read_excel(ANNEALING_SCHEDULE_XLS, sheet_name=1)
A = CubicSpline(df['s'], df['A(s) (GHz)'])
B = CubicSpline(df['s'], df['B(s) (GHz)'])

sigmax = np.array([[0, 1], [1, 0]])
sigmaz = np.diag([1, -1])

# compute omega for h and s_bar
sts = np.linspace(0.52, 0.80, 100)
hs = np.linspace(0, 1, 100)

X, Y = np.meshgrid(sts, hs)
Z = np.empty_like(X)

for ii, h in enumerate(hs):
    for jj, st in enumerate(sts):
        Z[ii, jj] = gap(st, h)

plt.figure('omega')
CS = plt.contour(X, Y, Z, 15, cmap='RdYlBu')
plt.pcolormesh(X, Y, Z, shading='nearest', cmap='RdBu')

plt.clabel(CS)
plt.colorbar()
plt.xlabel(r'$\bar{s}$')
plt.ylabel('h')
plt.title(r'$\omega$')

# scelgo 11 punti a omega fisso per 11 omega diversi
# fissati omega e st, trovo h
sts = np.linspace(0.58, 0.78, 11)
omegas = np.linspace(0.5, 5.5, 11)
hs = np.ndarray((11, 11))

h_tmp = np.linspace(0, 1, 5000)
omega_tmp = np.linspace(0, 1, 5000)

for i, st in enumerate(sts):
    for j, omega in enumerate(omegas):
        for k, h in enumerate(h_tmp):
            omega_tmp[k] = gap(st, h)
        hs[i, j] = h_tmp[np.argmin((omega_tmp - omega) ** 2)]
        if np.min((omega_tmp - omega) ** 2) > 0.1 ** 2:
            hs[i, j] = np.nan

hs = np.array([
    [0.03200, 0.19663, 0.31846, 0.43508, 0.54970, 0.66353, 0.77676, 0.88977,      1.,  np.nan,  np.nan],
    [0.07581, 0.19783, 0.30786, 0.41568, 0.52250, 0.62872, 0.73494, 0.84096, 0.94678,  np.nan,  np.nan],
    [0.08661, 0.19303, 0.29466, 0.39527, 0.49529, 0.59531, 0.69513, 0.79496, 0.89457, 0.99439,  np.nan],
    [0.08861, 0.18543, 0.28046, 0.37507, 0.46929, 0.56371, 0.65793, 0.75196, 0.84616, 0.94038,  np.nan],
    [0.08681, 0.17703, 0.26646, 0.35567, 0.44488, 0.53390, 0.62312, 0.71214, 0.80136, 0.89037, 0.97939],
    [0.08361, 0.16843, 0.25306, 0.33746, 0.42188, 0.50630, 0.59091, 0.67533, 0.75975, 0.84416, 0.92858],
    [0.07981, 0.16003, 0.24024, 0.32046, 0.40068, 0.48069, 0.56091, 0.64112, 0.72114, 0.80136, 0.88157],
    [0.07601, 0.15223, 0.22844, 0.30466, 0.38087, 0.45709, 0.53310, 0.60932, 0.68553, 0.76176, 0.83796],
    [0.07241, 0.14502, 0.21744, 0.29006, 0.36247, 0.43488, 0.50750, 0.57991, 0.65253, 0.72494, 0.79736],
    [0.06901, 0.13822, 0.20724, 0.27626, 0.34546, 0.41448, 0.48349, 0.55251, 0.62172, 0.69073, 0.75976],
    [0.06581, 0.13182, 0.19763, 0.26366, 0.32946, 0.39527, 0.46129, 0.52710, 0.59311, 0.65893, 0.72474]
])

for i, hs_row in enumerate(hs.transpose()):
    plt.scatter(sts, hs_row)
    
# experiment
with open(DATA_DIR / '7_115experiments_mean_spin.json', 'r') as f:
    exp_tmp = json.load(f)

with open(DATA_DIR / '7_115experiments_std_spin.json', 'r') as g:
    dexp_tmp = json.load(g)

exp = {}
dexp = {}
for key in exp_tmp.keys():
    param_couples = key.split(":")
    st = float(param_couples[0])
    h = float(param_couples[1])
    exp[(st, h)] = np.array(exp_tmp[key])
    dexp[(st, h)] = np.array(dexp_tmp[key])


# figure
plt.figure("therm", figsize=(15, 7.5))
plt.ylabel(r'$\langle \sigma_z \rangle $', fontsize=17)
plt.yticks(fontsize=12.5)
plt.xlabel('pause time (µs)', fontsize=15)
plt.xticks(fontsize=12.5)
plt.title(r"1 qubit thermalization", fontsize=17)

color_dict = {round(sts[i], 2): list(mcolors.TABLEAU_COLORS.keys())[i % 10] for i in range(len(omegas))}
marker_dict = {omegas[i]: list(Line2D.markers.keys())[i] for i in range(len(omegas))}
marker_dict[1.0] = '*'
marker_dict[4.0] = '8'
marker_dict[4.5] = 's'
marker_dict[5.0] = 'p'
marker_dict[5.5] = 'P'

ls = ['-', '--', '-.', '.']

x = list(range(0, 11))
for i, st in enumerate(sts):
    for num, h in enumerate(hs[i]):
        try:
            omega = round(2 * gap(st, h)) / 2
            alpha = 1 if num < 5 else 0.3
            st = round(st, 2)
            h = round(h, 4)
            plt.errorbar(x, exp[st, h], dexp[st, h], marker=marker_dict[omega], ls=ls[num % 3], c=color_dict[st], alpha=alpha)
        except ValueError:  # np.nan
            pass

prob1 = 1 / (1 + np.exp(48/14 * 0.5))
prob2 = 1 / (1 + np.exp(48/15 * 0.5))
# plt.plot([0, 10], [2 * prob1 - 1, 2 * prob1 - 1], c='C5')
# plt.plot([0, 10], [2 * prob2 - 1, 2 * prob2 - 1], c='C6')
# exp_ = exp[(0.65, 0.088)]
# plt.plot(x, 2 * (1 + exp_) / (1 + exp_[0]) - 1, marker='o', ls='-', c='C5')

c_handles = [Line2D([], [], color='w', markerfacecolor=color_dict[key], marker='o', markersize=10,
                    label='st=' + str(key)) for key in color_dict.keys()]
m_handles = [Line2D([], [], color='w', markerfacecolor=gray, marker=marker_dict[key], markersize=10,
                    label=r'$\omega$=' + str(key) + 'GHz') for key in marker_dict.keys()]

plt.legend(handles=c_handles+m_handles)
plt.tight_layout()

x = np.linspace(0, 50)
plt.figure('anneal schedule')
plt.plot(x, t_to_s(x, 48, 0.7))
plt.title('anneal schedule of experiment')
plt.xlabel('time (µs)')
plt.ylabel('s')


# simulation
sim_files = [x for x in DATA_DIR.glob('7_115kz/z_s*_h*_x*.pkl') if x.is_file()]
sim = {}

for file in sorted(sim_files):
    m = re.search(r'z_s(.*)_h(.*)_x(.*).pkl', file.name)
    st = float(m.group(1))
    h = float(m.group(2))
    kz = float(m.group(3))
    with open(file, 'rb') as fp:
        mean_spin = pickle.load(fp)
    sim[(st, h, kz)] = np.array(mean_spin)

# chi 2
chi2 = {}
for s_bar, h, kz in sim.keys():
    h
    chi2[(s_bar, h, kz)] = np.sum((exp[(s_bar, h)] - sim[(s_bar, h, kz)]) ** 2 / dexp[(s_bar, h)] ** 2)

simulated_experiments = set((st, h0) for (st, h0, kz) in sim.keys())

for i, st in enumerate(sts):
    for num, h in enumerate(hs[i]):
        st = round(st, 2)
        kz_list = []
        chi2_list = []
        for (st0, h0, kz) in sim.keys():
            if h == h0 and st == st0:
                kz_list.append(kz)
                chi2_list.append((chi2[st0, h0, kz]))

        if len(kz_list) != 0:
            chi2_func = CubicSpline(kz_list, chi2_list)
            kz_sample = np.linspace(min(kz_list), max(kz_list), 300)
            chi2_sample = [chi2_func(k) for k in kz_sample]
            # print('chi2 =', int(min(chi2_sample)), 'h:', h, 'st:', st)

            chi2_treshold = min(chi2_sample) + np.sqrt(2 * abs(min(chi2_sample)))
            k_opt = kz_sample[np.argmin(chi2_sample)]
            try:
                k_err = min(abs(chi2_func.solve(chi2_treshold) - k_opt)) * 2  # x2 artificioso
            except ValueError:
                k_err = 1
            omega = round(2 * gap(st, h)) / 2

            plt.figure('chi2')
            plt.scatter(kz_list, chi2_list, marker='o')
            plt.plot(kz_sample, chi2_sample, label=f'h={h}, st={st}')

            plt.figure('kz vs st')
            plt.errorbar(st, k_opt, k_err, c=color_dict[st], marker=marker_dict[omega], markersize=5.5, ecolor=gray, capsize=3, label=f'h={h}')

            plt.figure('kz vs omega')
            plt.errorbar(omega, k_opt, k_err, c=color_dict[st], marker=marker_dict[omega], markersize=5.5, ecolor=gray, capsize=3, label=f'h={h}')

plt.figure('chi2')
plt.xlabel(r'$k_z$')
plt.ylabel(r'$\chi^2$')
plt.ylabel(r'$\chi^2(\omega)$')
# plt.ylim((-10, 600))
# plt.legend()
plt.tight_layout()

plt.figure('kz vs st')
plt.xlabel(r'$s_t$')
plt.ylabel('$k_z$')
plt.title(r'$k_z(s_t)$')
plt.tight_layout()
plt.legend(handles=c_handles + m_handles)

plt.figure('kz vs omega')
plt.xlabel(r'$\omega$ (GHz)')
plt.ylabel('$k_z$')
plt.title(r'$k_z(\omega)$')
plt.tight_layout()
plt.legend(handles=c_handles + m_handles)

plt.show()
