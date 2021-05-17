import numpy as np
import pickle
import glob
import scipy.interpolate
from matplotlib import pyplot as plt

# load data (experiment)
biases1, spin_up1, dspin_up1 = pickle.load(open("data/results.pickle", "rb"))
biases2, spin_up2, dspin_up2 = pickle.load(open("data/results2.pickle", "rb"))

xx = np.linspace(-0.3, 0.3, 1000)

# load data (simulation) 1 qubit
temp1 = 13.8
h1, p1 = pickle.load(open(f"data/qu_lindblad_various_h{temp1}.pickle", "rb"))
funct1 = scipy.interpolate.CubicSpline(h1, p1)  # smooth curve

# grafico popolaz ground state vs h - 1 qubit
plt.figure('1 qubit temperature')
plt.scatter(xx, funct1(xx))
plt.errorbar(biases1, spin_up1, dspin_up1, marker='.', linestyle='')
plt.title('single qubit temperature')
plt.xlabel('h')
plt.ylabel(r'$P_\uparrow$')

# grafico popolaz ground state vs h - 2 qubits
plt.figure('2 qubits temperature')
plt.errorbar(biases2, spin_up2, dspin_up2, marker='.', linestyle='', label='2 qubits')
plt.errorbar(biases1, spin_up1, dspin_up1, marker='.', linestyle='', label='1 qubit', alpha=0.3)
plt.scatter(xx, funct1(xx), label=f'1 qubit, T = {temp1:.2f} mK', alpha=0.3)
# plot simulation for various temperature
for f in sorted(glob.glob("data/qu2_lindblad_various_h*.pickle")):
    h2, p2 = pickle.load(open(f, "rb"))
    temp2 = float(f[27:-7])
    funct2 = scipy.interpolate.CubicSpline(h2, p2)  # smooth curve
    plt.plot(xx, funct2(xx), label=f'2 qubits, T={temp2:.2f} mK')

    # print error on fit
    chi2 = 0
    for x, y, dy in zip(biases2, spin_up2, dspin_up2):
        chi2 += (y - funct2(x)) ** 2 / dy ** 2

    print(f'T={temp2:.2f}, fit err={chi2}')

plt.title('two qubits temperature')
plt.xlabel(r'$h_A$')
plt.ylabel(r'$P_\uparrow$')
plt.legend()

plt.show()
