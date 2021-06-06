import numpy as np
import json
# import scipy.interpolate
import matplotlib.pyplot as plt
from settings import DATA_DIR


cname = {
    "1 qubit": 'C0',
    "J=0.2": 'C1',
    "J=1": 'C2',
    "hB=0.1": 'C3'
}


def expected_population(sim_, T, k2, name):
    # load simulation w a given temperature and coupling
    simulation = [obj for obj in sim_ if obj["T"] == T and obj["k2"] == k2 and obj["name"] == name]

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

    if name == "1 qubit":
        oss_p = np.array([np.linalg.multi_dot([vector_p, rho, vector_p]) for rho in rho_list])
        oss_m = np.array([np.linalg.multi_dot([vector_m, rho, vector_m]) for rho in rho_list])

        return oss_p, oss_m

    oss_pp = np.array([np.linalg.multi_dot([vector_pp, rho, vector_pp]) for rho in rho_list])
    oss_pm = np.array([np.linalg.multi_dot([vector_pm, rho, vector_pm]) for rho in rho_list])
    oss_mp = np.array([np.linalg.multi_dot([vector_mp, rho, vector_mp]) for rho in rho_list])
    oss_mm = np.array([np.linalg.multi_dot([vector_mm, rho, vector_mm]) for rho in rho_list])

    return oss_pp, oss_pm, oss_mp, oss_mm


def chi2(sim_, T, k2, name, state_list):

    if name == "1 qubit":
        oss_p, oss_m = expected_population(sim_, T, k2, name)

        exp_p = state_list[0]
        exp_m = state_list[1]
        num = exp_p + exp_m
        dexp_p = np.sqrt((1 + exp_p * (num - exp_p)) / num ** 3)
        dexp_m = np.sqrt((1 + exp_m * (num - exp_m)) / num ** 3)

        chi2_p = (oss_p - exp_p) ** 2 / dexp_p ** 2
        chi2_m = (oss_m - exp_m) ** 2 / dexp_m ** 2

        return chi2_p, chi2_m

    oss_pp, oss_pm, oss_mp, oss_mm = expected_population(sim_, T, k2, name)

    exp_pp = state_list[0]
    exp_pm = state_list[1]
    exp_mp = state_list[2]
    exp_mm = state_list[3]
    num = exp_pp + exp_pm + exp_mp + exp_mm
    dexp_pp = np.sqrt((1 + exp_pp * (num - exp_pp)) / num ** 3)
    dexp_pm = np.sqrt((1 + exp_pm * (num - exp_pm)) / num ** 3)
    dexp_mp = np.sqrt((1 + exp_mp * (num - exp_mp)) / num ** 3)
    dexp_mm = np.sqrt((1 + exp_mm * (num - exp_mm)) / num ** 3)

    chi2_pp = (oss_pp - exp_pp) ** 2 / dexp_pp ** 2
    chi2_pm = (oss_pm - exp_pm) ** 2 / dexp_pm ** 2
    chi2_mp = (oss_mp - exp_mp) ** 2 / dexp_mp ** 2
    chi2_mm = (oss_mm - exp_mm) ** 2 / dexp_mm ** 2

    return chi2_pp, chi2_pm, chi2_mp, chi2_mm


def plot(sim_, T, k2):
    names_ = list(set([obj["name"] for obj in sim_]))
    for name in names_:
        if name == "1 qubit":
            oss_p, oss_m = expected_population(sim_, T, k2, name)
            fig_merit = oss_m
        else:
            oss_pp, oss_pm, oss_mp, oss_mm = expected_population(sim_, T, k2, name)
            fig_merit = oss_mp + oss_mm

            h = [obj["hA"] for obj in sim_ if obj["T"] == T and obj["k2"] == k2 and obj["name"] == name]
            plt.plot(h, oss_mp, color=cname[name], alpha=0.5, linestyle='--')
            plt.plot(h, oss_mm, color=cname[name], alpha=0.5, linestyle='-.')

        h = [obj["hA"] for obj in sim_ if obj["T"] == T and obj["k2"] == k2 and obj["name"] == name]
        plt.plot(h, fig_merit, color=cname[name])

        # funct = scipy.interpolate.CubicSpline(h, fig_merit)
        # x = np.linspace(min(h), max(h), 1000)
        # plt.plot(x, funct(x), label=f"T={T}, k2={k2}")

    plt.title(f"T={T}, k2={k2}")

    return


# EXPERIMENTS

with open(DATA_DIR / "multiple_experiments.json", "r") as read_file:
    data = json.load(read_file)

p = DATA_DIR.glob('multiple_experiments_simulation_T*_k*.json')
files = [x for x in p if x.is_file()]
sim = []
for file in files:
    with open(file, 'r') as fp:
        sim += json.load(fp)

names_2 = list(set([obj["name"] for obj in data]))
names_2.remove("1 qubit")

plt.figure()

# 1 qubit
data1 = [obj for obj in data if obj["name"] == "1 qubit"]

x1 = np.array([obj["hA"] for obj in data1])
states = np.array([obj["states"] for obj in data1])

state_p = states[:, 0]
state_m = states[:, 1]

down1 = state_m / (state_p + state_m)
std1 = np.sqrt(state_p * state_m / (state_m + state_p) ** 3)
plt.errorbar(x1, down1, std1, color=cname["1 qubit"], marker='.', linestyle='', label='1 qubit')

# 2 qubits
for name_obj in names_2:
    data2 = [obj for obj in data if obj["name"] == name_obj]

    x2 = np.array([obj["hA"] for obj in data2])
    hB = np.array([obj["hB"] for obj in data2])
    J = np.array([obj["J"] for obj in data2])

    states = np.array([obj["states"] for obj in data2])

    state_pp = states[:, 0]
    state_pm = states[:, 1]
    state_mp = states[:, 2]
    state_mm = states[:, 3]

    num_samplings = state_pp + state_pm + state_mp + state_mm
    down = (state_mp + state_mm) / num_samplings
    std = np.sqrt(state_mm * (num_samplings - state_mm) / num_samplings ** 3)

    plt.errorbar(x2, down, std, color=cname[name_obj], marker='.', linestyle='', label=f'hB={hB[0]}, J={J[0]}')

plot(sim, 14.0, 0.03)

plt.xlabel('hA')
plt.ylabel('spin A is up')
plt.legend()

# chi2
temperatures = [13.7, 13.9, 14.1]
k2s = [0.01, 0.03, 0.06]
X, Y = np.meshgrid(temperatures, k2s)
Z = np.zeros_like(X)

for i, x in enumerate(temperatures):
    for j, y in enumerate(k2s):
        # 1 qubit
        states = np.array([obj["states"] for obj in data if obj["name"] == "1 qubit"])
        state_list1 = [states[:, 0], states[:, 1]]
        Z[j, i] += np.sum(chi2(sim, x, y, "1 qubit", state_list1))

        # 2 qubits
        for name_obj in names_2:
            states = np.array([obj["states"] for obj in data if obj["name"] == name_obj])
            state_list2 = [states[:, 0], states[:, 1], states[:, 2], states[:, 3]]
            Z[j, i] += np.sum(chi2(sim, x, y, name_obj, state_list2))
        Z[j, i] = np.random.random()

plt.figure("chi2")
plt.pcolormesh(X, Y, Z, shading='nearest')  # shading='gouraud'
plt.colorbar()

plt.xlabel('T')
plt.ylabel('k2')
plt.title('Chi^2')

plt.show()
