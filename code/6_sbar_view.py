import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import re
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from scipy.interpolate import CubicSpline
from settings import DATA_DIR, ANNEALING_SCHEDULE_XLS


def t_to_s(t, pause_time, min_s):
    return (1 - min_s) * (1-t) * np.heaviside(1-t, 0) + min_s + (1 - min_s) * (t-pause_time-1) * np.heaviside(t-pause_time-1, 0)


gray = '0.4'
init_cutoff = 3
df = pd.read_excel(ANNEALING_SCHEDULE_XLS, sheet_name=1)
A = CubicSpline(df['s'], df['A(s) (GHz)'])
B = CubicSpline(df['s'], df['B(s) (GHz)'])

sigmax = np.array([[0, 1], [1, 0]])
sigmaz = np.diag([1, -1])

# experiment
exp_files = [x for x in DATA_DIR.glob('sbar_0.*_h*.pkl') if x.is_file()]
exp = {}
dexp = {}
for file in sorted(exp_files):
    m = re.search(r'sbar_(0.\d+)_h([\.\d]*).pkl', file.name)
    st = float(m.group(1))
    h = float(m.group(2))
    with open(file, 'rb') as fp:
        mean_spin, std_spin = pickle.load(fp)
    exp[(h, st)] = np.array(mean_spin) / h
    dexp[(h, st)] = np.array(std_spin) / h + 0.003


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


simxz_files = [x for x in DATA_DIR.glob('sim_xz/sbar_sim_xz_h*_s*_kz*_kx*.pkl') if x.is_file()]
simxz = {}

for file in sorted(simxz_files):
    m = re.search(r'sbar_sim_xz_h(.*)_s(.*)_kz(.*)_kx(.*).pkl', file.name)
    h = float(m.group(1))
    st = float(m.group(2))
    kz = float(m.group(3))
    kx = float(m.group(4))
    with open(file, 'rb') as fp:
        mean_spin = pickle.load(fp)
    simxz[(h, st, kz, kx)] = np.array(mean_spin)

# chi 2 xz
chi2xz = {}
for h, s_bar, kz, kx in simxz.keys():
    chi2xz[(h, s_bar, kz, kx)] = np.sum((exp[(h, s_bar)][init_cutoff:] - simxz[(h, s_bar, kz, kx)][init_cutoff:]) ** 2 / dexp[(h, s_bar)][init_cutoff:] ** 2)

h0s = sorted(set(h0 for (h0, st, kz, kx) in simxz.keys()))
sts = sorted(set(st for (h0, st, kz, kx) in simxz.keys()))

fig, axs = plt.subplots(len(sts), len(h0s), squeeze=False, sharey='all', sharex='all')
for i, h in enumerate(h0s):
    for j, st in enumerate(sts):
        kz_list = []
        kx_list = []
        for (h0, st0, kz, kx) in simxz.keys():
            if h == h0 and st == st0:
                kz_list.append(kz)
                kx_list.append(kx)
        kz_order = sorted(set(kz_list))
        kx_order = sorted(set(kx_list))

        X, Y = np.meshgrid(kx_order, kz_order)
        Z = np.empty_like(X)

        for ii, kx in enumerate(kx_order):
            for jj, kz in enumerate(kz_order):
                try:
                    Z[jj, ii] = chi2xz[(h, st, kz, kx)]
                except KeyError:
                    Z[jj, ii] = np.ma.masked

        ax = axs[j, i]
        ax.set_title(f'h={h}, s={st}')
        if i == 0:
            ax.set_ylabel('kz')
        if j == len(sts)-1:
            ax.set_xlabel('kx')
        try:
            c = ax.pcolormesh(X, Y, Z, shading='nearest', cmap='RdBu', norm=LogNorm())
            fig.colorbar(c, ax=ax)
        except Exception as e:
            print(e)

## YZ
simyz_files = [x for x in DATA_DIR.glob('sim_yz/sbar_sim_yz_h*_s*_kz*_ky*.pkl') if x.is_file()]
simyz = {}

for file in sorted(simyz_files):
    m = re.search(r'sbar_sim_yz_h(.*)_s(.*)_kz(.*)_ky(.*).pkl', file.name)
    h = float(m.group(1))
    st = float(m.group(2))
    kz = float(m.group(3))
    ky = float(m.group(4))
    with open(file, 'rb') as fp:
        mean_spin = pickle.load(fp)
    simyz[(h, st, kz, ky)] = np.array(mean_spin)

# chi 2 xz
chi2yz = {}
for h, s_bar, kz, ky in simyz.keys():
    chi2yz[(h, s_bar, kz, ky)] = np.sum((exp[(h, s_bar)][init_cutoff:] - simyz[(h, s_bar, kz, ky)][init_cutoff:]) ** 2 / dexp[(h, s_bar)][init_cutoff:] ** 2)

h0s = sorted(set(h0 for (h0, st, kz, ky) in simyz.keys()))
sts = sorted(set(st for (h0, st, kz, ky) in simyz.keys()))

fig, axs = plt.subplots(len(sts), len(h0s), squeeze=False, sharey='all', sharex='all')
for i, h in enumerate(h0s):
    for j, st in enumerate(sts):
        kz_list = []
        ky_list = []
        for (h0, st0, kz, ky) in simyz.keys():
            if h == h0 and st == st0:
                kz_list.append(kz)
                ky_list.append(ky)
        kz_order = sorted(set(kz_list))
        ky_order = sorted(set(ky_list))

        X, Y = np.meshgrid(ky_order, kz_order)
        Z = np.empty_like(X)

        for ii, ky in enumerate(ky_order):
            for jj, kz in enumerate(kz_order):
                try:
                    Z[jj, ii] = chi2yz[(h, st, kz, ky)]
                except KeyError:
                    Z[jj, ii] = np.ma.masked

        ax = axs[j, i]
        ax.set_title(f'h={h}, s={st}')
        if i == 0:
            ax.set_ylabel('kz')
        if j == len(sts)-1:
            ax.set_xlabel('ky')
        try:
            c = ax.pcolormesh(X, Y, Z, shading='nearest', cmap='RdBu', norm=LogNorm())
            fig.colorbar(c, ax=ax)
        except Exception as e:
            print(e)

# chi 2
chi2 = {}
for h, s_bar, kz in sim.keys():
    chi2[(h, s_bar, kz)] = np.sum((exp[(h, s_bar)][init_cutoff:] - sim[(h, s_bar, kz)][init_cutoff:]) ** 2 / dexp[(h, s_bar)][init_cutoff:] ** 2)

simulated_experiments = set((h0, st) for (h0, st, kz) in sim.keys())

c_dict = {
    1: 'C0',
    0.95: 'C1',
    0.9: 'C2'
}

m_dict = {
    0.66: '^',
    0.67: 'o',
    0.68: 's',
    0.69: 'D'
}

c_handles = [Line2D([], [], color='w', markerfacecolor=c_dict[key], marker='o', markersize=10, label='h=' + str(key)) for key in c_dict.keys()]
m_handles = [Line2D([], [], color='w', markerfacecolor=gray, marker=m_dict[key], markersize=10, label='st=' + str(key)) for key in m_dict.keys()]


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
    # print('chi2 =', int(min(chi2_sample)), 'h:', h, 'st:', st)

    chi2_treshold = min(chi2_sample) + np.sqrt(2 * abs(min(chi2_sample)))
    k_opt = kz_sample[np.argmin(chi2_sample)]
    k_err = min(abs(chi2_func.solve(chi2_treshold) - k_opt)) * 2  # x2 artificioso

    ham = A(st) * sigmax / 2 + h * B(st) * sigmaz / 2
    eigenvalues, eigenvectors = np.linalg.eigh(ham)
    omega = eigenvalues[1] - eigenvalues[0]

    plt.figure('chi2')
    plt.scatter(kz_list, chi2_list, marker='o')
    plt.plot(kz_sample, chi2_sample, label=f'h={h}, st={st}')

    plt.figure('kz vs st')
    plt.errorbar(st, k_opt, k_err, c=c_dict[h], ecolor=gray, capsize=3, marker=m_dict[st], markersize=7, label=f'h={h}')

    plt.figure('kz vs omega')
    plt.errorbar(omega, k_opt, k_err, c=c_dict[h], ecolor=gray, capsize=3, marker=m_dict[st], markersize=7, label=f'h={h}')

plt.figure('chi2')
plt.xlabel(r'$k_z$')
plt.ylabel(r'$\chi^2$')
plt.ylabel(r'$\chi^2(\omega)$')
plt.legend()
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

# figure
plt.figure("therm", figsize=(15, 7.5))
plt.ylabel(r'$\langle \sigma_z \rangle $', fontsize=17)
plt.yticks(fontsize=12.5)
plt.xlabel('pause time (µs)', fontsize=15)
plt.xticks(fontsize=12.5)
plt.title(r"1 qubit thermalization", fontsize=17)
x = list(range(0, 21, 2))

for i, (h, st) in enumerate(simulated_experiments):
    c_dict[(h, st)] = 'C' + str(i)
    plt.errorbar(x, exp[h, st], dexp[h, st], c=c_dict[(h, st)], marker='o', ls='-', label=f'h={h}, st={st}')

for key in []:
    plt.errorbar(x, exp[key], dexp[key], c='black', marker='o', ls='-', label=f'h={key[0]}, st={key[1]}')

for h, st, kz in sim.keys():
    plt.plot(x, sim[(h, st, kz)], c=c_dict[(h, st)], marker='.', ls='--', alpha=0.3)

plt.legend()
plt.tight_layout()

x = np.linspace(0, 50)
plt.figure('anneal schedule')
plt.plot(x, t_to_s(x, 48, 0.7))
plt.title('anneal schedule of experiment')
plt.xlabel('time (µs)')
plt.ylabel('s')


# KX same omega

fig, axs = plt.subplots(2, 2, squeeze=False, sharey='all', sharex='all')

parameters = [
    [(0.9, 0.68), (0.95, 0.66)],
    [(0.9, 0.69), (0.95, 0.67)],
    [(0.95, 0.68), (1, 0.66)],
    [(0.95, 0.69), (1, 0.67)]
    ]

omegas = [
    5.33,
    5.48,
    5.62,
    5.77
]
for i, couple_graphs in enumerate(parameters):
    # Z = np.zeros_like(X)
    Z = np.ones_like(X)
    for h, st in couple_graphs:
        for ii, kx in enumerate(kx_order):
            for jj, kz in enumerate(kz_order):
                # Z[jj, ii] += chi2xz[(h, st, kz, kx)]
                Z[jj, ii] *= chi2xz[(h, st, kz, kx)]

    ax = axs.flat[i]
    ax.set_title(f'omega={omegas[i]} GHz')
    c = ax.pcolormesh(X, Y, Z, shading='nearest', cmap='RdBu', norm=LogNorm())
    fig.colorbar(c, ax=ax)

    if i // 2 == 1:
        ax.set_xlabel('kx')
    if i % 2 == 0:
        ax.set_ylabel('kz')


plt.show()
