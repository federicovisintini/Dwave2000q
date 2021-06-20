import numpy as np
import matplotlib.pyplot as plt
import pickle
import re
from matplotlib.colors import LogNorm
from settings import DATA_DIR

logscale = False

# figure
plt.figure("therm", figsize=(15, 7.5))
plt.ylabel(r'$\langle E \rangle $', fontsize=17)
plt.yticks(fontsize=12.5)
plt.xlabel(r'$t (\mu s)$', fontsize=15)
plt.xticks(fontsize=12.5)
plt.title(r"1 qubit thermalization", fontsize=17)

# 1 qubit exp
files = [x for x in DATA_DIR.glob('coupling_st*.pkl') if x.is_file()]
exp = {}
for file in sorted(files):
    st = float(file.name[11: -4])
    with open(file, 'rb') as fp:
        exp[st] = pickle.load(fp)

x = np.linspace(0, 50, num=11)
for st in exp.keys():
    plt.plot(x, [1] + exp[st], marker='o', ls='')  # '--', alpha=0.2)


# 1 qubit sim
files = [x for x in DATA_DIR.glob('coupling_sim_st*_kx*_kz*.pkl') if x.is_file()]
sim = {}
for file in sorted(files):
    m = re.search(r'coupling_sim_st(0\.\d+)_kx([\.e\-\d]+)_kz(0\.\d+)\.pkl', file.name)
    st = float(m.group(1))
    kx = float(m.group(2))
    kz = float(m.group(3))
    with open(file, 'rb') as fp:
        sim[(st, kx, kz)] = pickle.load(fp)

st = 0.70
for kx, kz in [(2e-8, 0.02)]:
    pass
    # plt.plot(x, [1] + sim[(st, kx, kz)], marker='o', ls='-', label=f'kx: {kx}, kz: {kz}')

plt.plot(x, [1] + exp[st], marker='o', ls='--', label='original')
plt.legend()

# color figure
# create list of kx, kz
sts = np.unique([st for st, kx, kz in sim.keys()])
kxs = np.unique([kx for st, kx, kz in sim.keys()])
kzs = np.unique([kz for st, kx, kz in sim.keys()])
Nx = 3  # 5
Ny = 3  # 4

X, Y = np.meshgrid(kxs, kzs)

fig, axs = plt.subplots(Ny, Nx, sharex='all', sharey='all', squeeze=False, figsize=(15, 7.5))

xx = 0
yy = 0
for k, st in enumerate(sts):
    Z = np.empty_like(X)
    for i, kx in enumerate(kxs):
        for j, kz in enumerate(kzs):
            try:
                Z[j, i] = np.sum((np.array(sim[(st, kx, kz)]) - np.array(exp[st])) ** 2)
            except (ValueError, KeyError):
                Z[j, i] = np.ma.masked

    ax = axs[yy, xx]

    # title, log scale and labels
    ax.set_title('st' + str(st))
    if logscale:
        ax.set_xscale('log')
        ax.set_xlim((min(kxs) / 2, max(kxs) * 1.45))
        ax.set_yscale('log')
    if xx == (Nx-1):
        ax.set_xlabel('kx')
    if yy == (Ny-1):
        ax.set_ylabel('kz')

    z_min, z_max = abs(Z).min(), abs(Z).max()
    c = ax.pcolormesh(X, Y, Z, shading='nearest', cmap='RdBu', norm=LogNorm())  # z_min, z_max
    fig.colorbar(c, ax=ax)

    xx += 1
    if xx == Nx:
        yy += 1
        xx = 0

plt.tight_layout()
plt.show()
