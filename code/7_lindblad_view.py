import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import json
import re
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from scipy.interpolate import CubicSpline
from settings import DATA_DIR, ANNEALING_SCHEDULE_XLS, STS, HS_LIST


def t_to_s(t, pause_time, min_s):
    return (1 - min_s) * (1-t) * np.heaviside(1-t, 0) + min_s + (1 - min_s) *\
           (t-pause_time-1) * np.heaviside(t-pause_time-1, 0)


def gap(st, h):
    hamiltonian = A(st) * sigmax / 2 + h * B(st) * sigmaz / 2
    eigenvals, eigenvects = np.linalg.eigh(hamiltonian)
    energy_gap = eigenvals[1] - eigenvals[0]
    return energy_gap


initial_cutoff = 1
gray = '0.4'
df = pd.read_excel(ANNEALING_SCHEDULE_XLS, sheet_name=1)
A = CubicSpline(df['s'], df['A(s) (GHz)'])
B = CubicSpline(df['s'], df['B(s) (GHz)'])

sigmax = np.array([[0, 1], [1, 0]])
sigmaz = np.diag([1, -1])

# compute omega for h and s_bar
sts = np.linspace(0.52, 0.78, 100)
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

# scelgo 4 punti a omega fisso per 6/7 omega diversi
# STS, HS_LIST in settings
for h in np.linspace(0, 1, 1000):
    st = 0.55
    if 0.49 < gap(st, h) < 0.51:
        print(h, gap(st, h))

for i, hs in enumerate(np.array(HS_LIST).transpose()):
    plt.scatter(STS, hs)
    for st, h in zip(STS, hs):
        print(gap(st, h))
    print()

# experiment
with open(DATA_DIR / '7_20experiments_mean_spin.json', 'r') as f:
    exp_tmp = json.load(f)

with open(DATA_DIR / '7_20experiments_std_spin.json', 'r') as g:
    dexp_tmp = json.load(g)

exp = {}
dexp = {}
for key in exp_tmp.keys():
    param_couples = key.split(":")
    st = float(param_couples[0])
    h = float(param_couples[1])
    exp[(st, h)] = np.array(exp_tmp[key])
    dexp[(st, h)] = np.array(dexp_tmp[key]) + 1e-5


# figure
plt.figure("therm", figsize=(15, 7.5))
plt.ylabel(r'$\langle \sigma_z \rangle $', fontsize=17)
plt.yticks(fontsize=12.5)
plt.xlabel('pause time (µs)', fontsize=15)
plt.xticks(fontsize=12.5)
plt.title(r"1 qubit thermalization", fontsize=17)

color_dict = {
    0.60: 'tab:blue',
    0.65: 'tab:orange',
    0.70: 'tab:green',
    0.75: 'tab:red',
    # 0.55: 'tab:gray'
}
marker = ['o', 'v', 's', '^', 'D', '<', 'X']
ls = ['-', '--', '-.']
omegas = [0.5, 1, 1.5, 2, 3, 4, 5]
m_dict = {omegas[i]: marker[i] for i in range(len(marker))}

x = list(range(0, 11))
for key in exp.keys():
    st, h = key
    num = np.argwhere(np.array(HS_LIST) == h)[0][1]  # [0, 5]
    alpha = 1 if num < 3 else 0.3
    plt.errorbar(x, exp[key], dexp[key], marker=marker[num], ls=ls[num % 3], c=color_dict[st], alpha=alpha)


prob1 = 1 / (1 + np.exp(48/14 * 0.5))
prob2 = 1 / (1 + np.exp(48/15 * 0.5))
# plt.plot([0, 10], [2 * prob1 - 1, 2 * prob1 - 1], c='C5')
# plt.plot([0, 10], [2 * prob2 - 1, 2 * prob2 - 1], c='C6')
# exp_ = exp[(0.65, 0.088)]
# plt.plot(x, 2 * (1 + exp_) / (1 + exp_[0]) - 1, marker='o', ls='-', c='C5')

c_handles = [Line2D([], [], color='w', markerfacecolor=color_dict[key], marker='o', markersize=10, label='st=' + str(key)) for key in color_dict.keys()]
m_handles = [Line2D([], [], color='w', markerfacecolor=gray, marker=m_dict[key], markersize=10, label='$\omega$=' + str(key) + 'GHz') for key in m_dict.keys()]

plt.legend(handles=c_handles+m_handles)
plt.tight_layout()

x = np.linspace(0, 50)
plt.figure('anneal schedule')
plt.plot(x, t_to_s(x, 48, 0.7))
plt.title('anneal schedule of experiment')
plt.xlabel('time (µs)')
plt.ylabel('s')


# simulation
sim_files = [x for x in DATA_DIR.glob('7_kz/z_s*_h*_kz*.pkl') if x.is_file()]
sim = {}

for file in sorted(sim_files):
    m = re.search(r'z_s(.*)_h(.*)_kz(.*).pkl', file.name)
    st = float(m.group(1))
    h = float(m.group(2))
    kz = float(m.group(3))
    with open(file, 'rb') as fp:
        mean_spin = pickle.load(fp)
    sim[(st, h, kz)] = np.array(mean_spin)


simxz_files = [x for x in DATA_DIR.glob('7_kzkx/zx_s*_h*_kz*_kx*.pkl') if x.is_file()]
simxz = {}

for file in sorted(simxz_files):
    m = re.search(r'zx_s(.*)_h(.*)_kz(.*)_kx(.*).pkl', file.name)
    st = float(m.group(1))
    h = float(m.group(2))
    kz = float(m.group(3))
    kx = float(m.group(4))
    with open(file, 'rb') as fp:
        mean_spin = pickle.load(fp)
    simxz[(st, h, kz, kx)] = np.array(mean_spin)


# chi 2 xz
chi2xz = {}
for s_bar, h, kz, kx in simxz.keys():
    chi2xz[(s_bar, h, kz, kx)] = np.sum((exp[(s_bar, h)] - simxz[(s_bar, h, kz, kx)]) ** 2 / dexp[(s_bar, h)] ** 2)

fig, axs = plt.subplots(5, 4, squeeze=False, sharey='all', sharex='all')
for i, (st, hs) in enumerate(zip(STS, HS_LIST)):
    for j, h in enumerate(hs[2:]):
        kz_list = []
        kx_list = []
        for (st0, h0, kz, kx) in simxz.keys():
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
                    Z[jj, ii] = chi2xz[(st, h, kz, kx)]
                except KeyError:
                    Z[jj, ii] = np.ma.masked
        if Z.shape == (0, 0):
            break
        ax = axs[j, i]
        ax.set_title(f'h={h}, s={st}')
        if i == 0:
            ax.set_ylabel('kz')
        if j == len(STS):
            ax.set_xlabel('kx')
        c = ax.pcolormesh(X, Y, Z, shading='nearest', cmap='RdBu', norm=LogNorm())
        # fig.colorbar(c, ax=ax)


# chi 2
chi2 = {}
for s_bar, h, kz in sim.keys():
    chi2[(s_bar, h, kz)] = np.sum((exp[(s_bar, h)][initial_cutoff:] - sim[(s_bar, h, kz)][initial_cutoff:]) ** 2 / dexp[(s_bar, h)][initial_cutoff:] ** 2)

simulated_experiments = set((st, h0) for (st, h0, kz) in sim.keys())

for st, h in simulated_experiments:
    kz_list = []
    chi2_list = []
    for (st0, h0, kz) in sim.keys():
        if h == h0 and st == st0:
            kz_list.append(kz)
            chi2_list.append((chi2[st0, h0, kz]))

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
    omega = gap(st, h)
    num = np.argwhere(np.array(HS_LIST) == h)[0][1]  # [0, 5]

    plt.figure('chi2')
    plt.scatter(kz_list, chi2_list, marker='o')
    plt.plot(kz_sample, chi2_sample, label=f'h={h}, st={st}')

    plt.figure('kz vs st')
    plt.errorbar(st, k_opt, k_err, c=color_dict[st], marker=marker[num], markersize=5.5, ecolor=gray, capsize=3, label=f'h={h}')

    plt.figure('kz vs omega')
    plt.errorbar(omega, k_opt, k_err, c=color_dict[st], marker=marker[num], markersize=5.5, ecolor=gray, capsize=3, label=f'h={h}')

plt.figure('chi2')
plt.xlabel(r'$k_z$')
plt.ylabel(r'$\chi^2$')
plt.ylabel(r'$\chi^2(\omega)$')
plt.ylim((-10, 600))
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

# KX same omega
fig, axs = plt.subplots(2, 4, squeeze=False, sharey='all', sharex='all')
axs.flat[7].set_visible(False)

for i, omega in enumerate(omegas):
    # Z = np.zeros_like(X)
    Z = np.ones_like(X)
    for st, h in zip(STS, np.array(HS_LIST)[:, i]):
        if h == h:
            for ii, kx in enumerate(kx_order):
                for jj, kz in enumerate(kz_order):
                    # Z[jj, ii] += chi2xz[(st, h, kz, kx)]
                    Z[jj, ii] *= chi2xz[(st, h, kz, kx)]

    ax = axs.flat[i]
    ax.set_title(f'omega={omegas[i]} GHz')
    c = ax.pcolormesh(X, Y, Z, shading='nearest', cmap='RdBu', norm=LogNorm())
    # fig.colorbar(c, ax=ax)

    if i // 2 == 1:
        ax.set_xlabel('kx')
    if i % 2 == 0:
        ax.set_ylabel('kz')

plt.show()
