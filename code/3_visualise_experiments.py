import numpy as np
import json
import scipy.interpolate
import matplotlib.pyplot as plt
from settings import DATA_DIR
from matplotlib.colors import LogNorm

# temperatures = [13.6, 14.0, 14.2, 14.4, 14.6, 14.8, 15.0, 15.2, 15.4, 15.6, 15.8, 16.0, 16.4]
# k2s = [0.004, 0.005, 0.006, 0.008, 0.012, 0.020, 0.028, 0.04, 0.06, 0.09, 0.12]
temp_plot = 15.0
k2_plot = 0.006

cname = {
    "1 qubit": 'C0',
    "J=0.2": 'C1',
    "J=1": 'C2',
    "hB=0.1": 'C3'
}

err_sum1 = 10
err_sum2 = 10
interpolate = False
logscale = True

# EXPERIMENTS
temperatures = []
k2s = []

with open(DATA_DIR / "multiple_experiments.json", "r") as read_file:
    data = json.load(read_file)

files = [x for x in DATA_DIR.glob('multiple_experiments_simulation_T*_k*.json') if x.is_file()]
simulations_glob_list = []
for file in files:
    temperatures.append(float(file.name[33:37]))
    k2s.append(float(file.name[39:-5]))
    with open(file, 'r') as fp:
        simulations_glob_list += json.load(fp)

temperatures = np.unique(temperatures)
k2s = np.unique(k2s)

name1 = "1 qubit"
names_2 = ['hB=0.1', 'J=0.2', 'J=1']


def expected_population(T, k2, name_):
    # load simulation w a given temperature and coupling
    simulation = [obj for obj in simulations_glob_list if obj["T"] == T and obj["k2"] == k2 and obj["name"] == name_]

    # select which curve to sample and compute its density matrix
    # rho_list = [np.array(obj["rho_real"]) + 1j * np.array(obj["rho_imag"]) for obj in simulation]
    rho_list = [np.array(obj["rho_real"]) for obj in simulation]

    # define observables for 1 and 2 qubits simulation
    vector_p = np.array([1, 0])
    vector_m = np.array([0, 1])

    vector_pp = np.kron(vector_p, vector_p)
    vector_pm = np.kron(vector_p, vector_m)
    vector_mp = np.kron(vector_m, vector_p)
    vector_mm = np.kron(vector_m, vector_m)

    if name_ == name1:
        oss_p = np.array([np.linalg.multi_dot([vector_p, rho, vector_p]) for rho in rho_list])
        oss_m = np.array([np.linalg.multi_dot([vector_m, rho, vector_m]) for rho in rho_list])

        return oss_p, oss_m

    oss_pp = np.array([np.linalg.multi_dot([vector_pp, rho, vector_pp]) for rho in rho_list])
    oss_pm = np.array([np.linalg.multi_dot([vector_pm, rho, vector_pm]) for rho in rho_list])
    oss_mp = np.array([np.linalg.multi_dot([vector_mp, rho, vector_mp]) for rho in rho_list])
    oss_mm = np.array([np.linalg.multi_dot([vector_mm, rho, vector_mm]) for rho in rho_list])

    return oss_pp, oss_pm, oss_mp, oss_mm


def chi2(T, k2, name_, state_list):

    if name_ == name1:
        exp_p, exp_m = expected_population(T, k2, name_)

        oss_p = state_list[0]
        oss_m = state_list[1]
        num = oss_p + oss_m

        chi2_p = (oss_p - exp_p * num) ** 2 / (exp_p * num)
        chi2_m = (oss_m - exp_m * num) ** 2 / (exp_m * num)

        # return np.sum(chi2_p), np.sum(chi2_m)
        return np.sum(chi2_m)

    exp_pp, exp_pm, exp_mp, exp_mm = expected_population(T, k2, name_)

    oss_pp = state_list[0]
    oss_pm = state_list[1]
    oss_mp = state_list[2]
    oss_mm = state_list[3]
    num = oss_pp + oss_pm + oss_mp + oss_mm

    chi2_pp = (oss_pp - exp_pp * num) ** 2 / (exp_pp * num)
    chi2_pm = (oss_pm - exp_pm * num) ** 2 / (exp_pm * num)
    chi2_mp = (oss_mp - exp_mp * num) ** 2 / (exp_mp * num)
    chi2_mm = (oss_mm - exp_mm * num) ** 2 / (exp_mm * num)
    chi2_fig = ((oss_mp + oss_mm) - (exp_mp + exp_mm) * num) ** 2 / ((exp_mp + exp_mm) * num)

    # return np.sum(chi2_pp), np.sum(chi2_pm), np.sum(chi2_mp), np.sum(chi2_mm)
    return np.sum(chi2_fig)


def plot_simulation(T, k2, alpha=1., name_plot=None):
    plt.figure("main")
    plt.title(f"T={T}, k2={k2}")

    # 1 qubit plot
    if name_plot == name1 or name_plot is None:
        oss_p, oss_m = expected_population(T, k2, name1)
        fig_merit = oss_m

        h = [obj["hA"] for obj in simulations_glob_list if obj["T"] == T and obj["k2"] == k2 and obj["name"] == name1]
        if interpolate:
            funct = scipy.interpolate.CubicSpline(h, fig_merit)
            x = np.linspace(min(h), max(h), 1000)
            plt.plot(x, funct(x), color=cname[name1], label=name1 + f"; $T={T}$mK, $k^2={k2}$")
        else:
            plt.plot(h, fig_merit, color=cname[name1], label=name1 + f"; $T={T}$mK, $k^2={k2}$")

        # residui
        state_p_, state_m_ = state_population(name1)
        num_samplings_ = state_p_ + state_m_
        fig_merit_data = state_m_ / num_samplings_
        std_ = np.sqrt((err_sum1 ** 2 + state_m_ * (num_samplings_ - state_m_)) / num_samplings_ ** 3)

        plt.figure("residui")
        plt.plot(h, fig_merit - fig_merit_data, color=cname[name1], marker='.',
                 linestyle='--', label=name1 + f"; $T={T}mk$, $k^2={k2}$", alpha=alpha)
        plt.title(f"T={T}, k2={k2}")
        plt.plot(h, np.zeros(len(h)), linestyle='--', color='black')

    # 2 qubit plot
    for name_ in names_2:
        if name_plot == name_ or name_plot is None:
            oss_pp, oss_pm, oss_mp, oss_mm = expected_population(T, k2, name_)
            fig_merit = oss_mp + oss_mm

            plt.figure("main")
            h = [obj["hA"] for obj in simulations_glob_list if obj["T"] == T and obj["k2"] == k2 and obj["name"] == name_]
            if interpolate:
                funct = scipy.interpolate.CubicSpline(h, fig_merit)
                x = np.linspace(min(h), max(h), 1000)
                plt.plot(x, funct(x), color=cname[name_], label=name_ + f"; $T={T}$mK, $k^2={k2}$")
            else:
                plt.plot(h, fig_merit, color=cname[name_], label=name_ + f"; $T={T}$mK, $k^2={k2}$")

            # components
            plt.plot(h, oss_mp, color=cname[name_], alpha=0.5, linestyle='--')
            plt.plot(h, oss_mm, color=cname[name_], alpha=0.5, linestyle='-.')

            # residui
            state_pp_, state_pm_, state_mp_, state_mm_ = state_population(name_)
            num_samplings_ = state_pp_ + state_pm_ + state_mp_ + state_mm_
            fig_merit_data = (state_mp_ + state_mm_) / num_samplings_
            std_ = np.sqrt(err_sum2 ** 2 / num_samplings_ ** 3 + fig_merit_data * (1 - fig_merit_data) / num_samplings_)

            plt.figure("residui")
            plt.plot(h, fig_merit - fig_merit_data, color=cname[name_], marker='.',
                         linestyle='--', label=name_ + f"; $T={T}$mK, $k^2={k2}$", alpha=alpha)

    return


def state_population(name_):
    actual_states = np.array([obj["states"] for obj in data if obj["name"] == name_])

    if name_ == name1:
        actual_state_p = actual_states[:, 0]
        actual_state_m = actual_states[:, 1]

        return actual_state_p, actual_state_m

    actual_state_pp = actual_states[:, 0]
    actual_state_pm = actual_states[:, 1]
    actual_state_mp = actual_states[:, 2]
    actual_state_mm = actual_states[:, 3]

    return actual_state_pp, actual_state_pm, actual_state_mp, actual_state_mm


# plot 1 qubit
x1 = [obj["hA"] for obj in data if obj["name"] == name1]
state_p, state_m = state_population(name1)

down1 = state_m / (state_p + state_m)
std1 = np.sqrt((err_sum1 ** 2 + state_p * state_m) / (state_m + state_p) ** 3)

plt.figure("main")
plt.errorbar(x1, down1, std1, color=cname["1 qubit"], marker='.', linestyle='', label='1 qubit')

# plot 2 qubits
for name in names_2:
    data2 = [obj for obj in data if obj["name"] == name]

    x2 = np.array([obj["hA"] for obj in data2])
    hB = np.array([obj["hB"] for obj in data2])
    J = np.array([obj["J"] for obj in data2])

    state_pp, state_pm, state_mp, state_mm = state_population(name)

    num_samplings = state_pp + state_pm + state_mp + state_mm
    down = (state_mp + state_mm) / num_samplings
    std = np.sqrt(err_sum2 ** 2 / num_samplings ** 3 + down * (1 - down) / num_samplings)

    plt.figure("main")
    plt.errorbar(x2, down, std, color=cname[name], marker='.', linestyle='', label=f'hB={hB[0]}, J={J[0]}')

# chi2
X, Y = np.meshgrid(temperatures, k2s)
Z = np.ma.array(np.empty_like(X))
Z0 = np.ma.array(np.empty_like(X))
Z1 = np.ma.array(np.empty_like(X))
Z2 = np.ma.array(np.empty_like(X))

for i, x in enumerate(temperatures):
    for j, y in enumerate(k2s):
        try:
            Z[j, i] = chi2(x, y, name1, state_population(name1)) / 29
        except ValueError as e:
            Z[j, i] = np.ma.masked

        try:
            Z0[j, i] = chi2(x, y, names_2[0], state_population(names_2[0])) / 29
            Z1[j, i] = chi2(x, y, names_2[1], state_population(names_2[1])) / 29
            Z2[j, i] = chi2(x, y, names_2[2], state_population(names_2[2])) / 29
        except ValueError:
            Z0[j, i] = np.ma.masked
            Z1[j, i] = np.ma.masked
            Z2[j, i] = np.ma.masked

# sum chi2
Zsum = (Z + Z0 + Z1 + Z2) / 4
plt.figure("chi2 sum")
plt.title('Chi^2 ridotto medio')
plt.xlabel('T')
plt.ylabel('k2')
plt.pcolormesh(X, Y, Zsum, shading='nearest', norm=LogNorm())  # shading='gouraud'
plt.colorbar()
if logscale:
    plt.yscale('log')

# multiple chi2
plt.figure("chi2 multiple")
plt.title('Chi^2 ridotto')

plt.subplot(221)
plt.pcolormesh(X, Y, Z, shading='nearest', norm=LogNorm())  # shading='gouraud'
plt.title('1 qubit')
plt.ylabel('k2')
plt.colorbar()
if logscale:
    plt.yscale('log')

plt.subplot(222)
plt.pcolormesh(X, Y, Z0, shading='nearest', norm=LogNorm())  # shading='gouraud'
plt.title(names_2[0])
plt.colorbar()
if logscale:
    plt.yscale('log')

plt.subplot(223)
plt.pcolormesh(X, Y, Z1, shading='nearest', norm=LogNorm())  # shading='gouraud'
plt.title(names_2[1])
plt.xlabel('T')
plt.ylabel('k2')
plt.colorbar()
if logscale:
    plt.yscale('log')

plt.subplot(224)
plt.pcolormesh(X, Y, Z2, shading='nearest', norm=LogNorm())  # shading='gouraud'
plt.title(names_2[2])
plt.xlabel('T')
plt.colorbar()
if logscale:
    plt.yscale('log')


# plot
# plot_simulation(T=temp_plot, k2=k2_plot)

k2_index, t_index = np.where(Z == Z.min())
plot_simulation(T=temperatures[t_index[0]], k2=k2s[k2_index[0]], name_plot=name1)
print(name1, ': ', Z.min())

k2_index, t_index = np.where(Z0 == Z0.min())
plot_simulation(T=temperatures[t_index[0]], k2=k2s[k2_index[0]], name_plot=names_2[0])
print(names_2[0], ': ', Z0.min())

k2_index, t_index = np.where(Z1 == Z1.min())
plot_simulation(T=temperatures[t_index[0]], k2=k2s[k2_index[0]], name_plot=names_2[1])
print(names_2[1], ': ', Z1.min())

k2_index, t_index = np.where(Z2 == Z2.min())
plot_simulation(T=temperatures[t_index[0]], k2=k2s[k2_index[0]], name_plot=names_2[2])
print(names_2[2], ': ', Z2.min())


plt.figure("main")
plt.xlabel('hA')
plt.ylabel('spin A is up')
plt.legend()

plt.figure("residui")
plt.xlabel('hA')
plt.ylabel('spin A is up - residui')
plt.legend()

plt.show()
