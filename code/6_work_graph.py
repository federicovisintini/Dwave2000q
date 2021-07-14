import numpy as np
import pandas as pd
import scipy.interpolate
import scipy.stats
from matplotlib import pyplot as plt
from settings import *

s_low = 0.65
s_high = 0.67
h = 0.15

ta = 50 + 1
tb = ta + 1
tc = tb + 100

# annealing functions
df = pd.read_excel(ANNEALING_SCHEDULE_XLS, sheet_name=1)
A = scipy.interpolate.CubicSpline(df['s'], df['A(s) (GHz)'])
B = scipy.interpolate.CubicSpline(df['s'], df['B(s) (GHz)'])
beta = 48 / 13.5  # 1 / GHz


def free_energy(E):
    return - np.log(np.exp(- beta * E) + 1) / beta


expected_extracted_work = free_energy(h * B(s_low)) - free_energy(h * B(s_high))
max_work = h * (B(s_high) - B(s_low)) / 2
min_work = - h * (B(s_high) - B(s_low)) / 2
prob = 1 / (1 + np.exp(beta * h * B(s_low)))

print(f'-∆F = {expected_extracted_work:.4f} GHz')
print(f'W_max = {max_work:.4f} GHz')
print(f'W_min = {min_work:.4f} GHz')

print(f'\nT = {48/beta} GHz')
print('p =', prob, '\n')

x = np.linspace(0, 1, 1000)

plt.figure('annealing functions')
plt.plot(x, A(x), label='A(s) GHz')
plt.plot(x, h * B(x), label=f'{h} * B(s) GHz')
plt.plot([0, 1], [1 / beta] * 2, ls='--', label='E=kT')
plt.plot([s_low, s_low], [0, 10], ls='--', c='black', alpha=0.5)
plt.plot([s_high, s_high], [0, 10], ls='--', c='black', alpha=0.5)
plt.title('anneling functions')
plt.xlabel('s')
plt.ylabel('Energy (GHZ)')
plt.tight_layout()
plt.legend()

plt.figure('rapporti')
plt.yscale('log')
plt.plot(x, A(x) / B(x) / h, label=f'A(s) / {h} * B(s)')
plt.plot(x, 1 / (h * B(x) * beta), label=f'kT / {h} * B(s)')
plt.plot([0, 1], [1, 1], ls='--', label='equal energy')
plt.plot([s_low, s_low], [5e-8, 5e2], ls='--', c='black', alpha=0.5)
plt.plot([s_high, s_high], [5e-8, 5e2], ls='--', c='black', alpha=0.5)
plt.xlabel('s')
plt.tight_layout()
plt.legend()

plt.figure('expected work')
# plt.yscale('log')
plt.ylim((0, 1.05))
plt.plot([max_work, max_work], [0, prob], c='C1', label=r'$W_{max}$')
plt.plot([min_work, min_work], [0, 1-prob], c='C2', label=r'$W_{min}$')
plt.plot([expected_extracted_work, expected_extracted_work], [0, 1], c='C0', label=r'$-\Delta F$')
plt.xlabel('W')
plt.ylabel('p(W)')
plt.title('expected probability of work extraction distribution')
plt.legend()

plt.figure('free energy vs energy')
energy = np.linspace(0.4, 10, 1000)
plt.xscale('log')
plt.plot(energy, free_energy(energy))
plt.plot([h * B(s_low), h * B(s_low)], [-0.06, 0], ls='--', c='black', alpha=0.5)
plt.plot([h * B(s_high), h * B(s_high)], [-0.06, 0], ls='--', c='black', alpha=0.5)
plt.xlabel('energy: E (GHz)')
plt.ylabel('Free energy: F (GHz)')
plt.title('free energy vs energy')

plt.figure('free energy vs s')
x = np.linspace(0.4, 1, 1000)
plt.plot(x, free_energy(h * B(x)), label='F(E): free energy')
plt.plot(x, h * B(x), label='E: energy')
plt.plot([s_low, s_low], [-0.1, 1], ls='--', c='black', alpha=0.5)
plt.plot([s_high, s_high], [-0.1, 1], ls='--', c='black', alpha=0.5)
plt.xlabel('s')
plt.ylabel('F (GHz)')
plt.title('free energy vs s')
plt.legend()

plt.figure('annealing schedule')
plt.plot([0, 1, ta], [1, s_low, s_low], c='C0', ls='--')
plt.plot([ta, tb, tc], [s_low, s_high, s_high], c='C0', marker='.')
plt.plot([tc, tc+1], [s_high, 1], c='C0', ls='--')
plt.scatter(ta, s_low, c='C0', label='A', zorder=3)
plt.scatter(tb, s_high, c='C1', label='B', zorder=3)
plt.scatter(tc, s_high, c='C2', label='C', zorder=3)
plt.legend()
plt.xlabel('t (µs)')
plt.ylabel('s')
plt.title('annealing schedule')

plt.figure('invasive measures')
plt.plot([0, 1, 2], [1, s_high, 1])
plt.plot([0, 1, 2], [1, s_low, 1])
plt.title('measure quench')
plt.xlabel('t (µs)')
plt.ylabel('s')

fig, axs = plt.subplots(3, 1)
ax = axs[0]
ax.plot([0, 1, ta, ta+1], [1, s_low, s_low, 1])
ax.scatter(ta, s_low, label='A', zorder=3)
ax.legend(loc='upper center')
ax.set_ylabel('s')
ax.set_title('measure A: $p(x_a)$')
ax.set_ylim([0.6, 1.05])

ax = axs[1]
ax.plot([0, 1, 2, 3], [1, s_low, s_high, 1])
ax.scatter(1, s_low, label='A', zorder=3)
ax.scatter(2, s_high, c='C1', label='B', zorder=3)
ax.legend(loc='upper center')
ax.set_title('measure B: $p(x_b|x_a)$')
ax.set_ylim([0.6, 1.05])

ax = axs[2]
ax.plot([0, 1, tc - tb + 1, tc - tb + 2], [1, s_high, s_high, 1])
ax.scatter(1, s_high, c='C1', label='B', zorder=3)
ax.scatter(tc - tb + 1, s_high, c='C2', label='C', zorder=3)
ax.legend(loc='upper center')
ax.set_xlabel('t (µs)')
ax.set_ylabel('s')
ax.set_title('measure C: $p(x_c|x_b)$')
ax.set_ylim([0.6, 1.05])

plt.tight_layout()

# RESULTS
# s_low = 0.65
# s_high = 0.67
# h = 0.13
QLp = 0.6880; dQLp = 0.0014
QHp = 0.8569; dQHp = 0.0008
Ap = -0.7471; dAp = 0.0011
Am = -0.7408; dAm = 0.0011
Bp = -0.5582; dBp = 0.0014
Bm = -0.7587; dBm = 0.0011
Cp = -0.7450; dCp = 0.0010
Cm = -0.7514; dCm = 0.0010

# s_low = 0.65
# s_high = 0.67
# h = 0.15
QLp = 0.92272; dQLp = 0.00058
QHp = 0.97065; dQHp = 0.00037
Ap = -0.89485; dAp = 0.00069
Am = -0.89301; dAm = 0.00071
Bp = -0.12788; dBp = 0.00142
Bm = -0.93038; dBm = 0.00054
Cp = -0.89974; dCp = 0.00068
Cm = -0.90245; dCm = 0.00074


energy_high = h * B(s_high) / 2
energy_low = h * B(s_low) / 2

work_pp = + energy_high - energy_low
work_mm = - energy_high + energy_low
work_pm = + energy_high + energy_low
work_mp = - energy_high - energy_low

p_pp = (1 + Am) * (1 + Bp) / 4
p_pm = (1 + Am) * (1 - Bp) / 4
p_mp = (1 - Am) * (1 + Bm) / 4
p_mm = (1 - Am) * (1 - Bm) / 4

plt.figure('work')
plt.plot([work_pp, work_pp], [0, p_pp], label='pp', c='C1')
plt.plot([work_mm, work_mm], [0, p_mm], label='mm', c='C2')
plt.plot([work_pm, work_pm], [0, p_pm], label='pm', c='C3')
plt.plot([work_mp, work_mp], [0, p_mp], label='mp', c='C4')
plt.ylim((0, 1.05))
plt.xlabel('W')
plt.ylabel('p(W)')
plt.title('probability of work extraction distribution')
plt.legend()

plt.show()
