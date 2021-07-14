import numpy as np
import pandas as pd
import scipy.interpolate
import scipy.stats
from matplotlib import pyplot as plt
from settings import *

s_low = 0.67
s_high = 0.75
h = 0.15

ta = 100 + 1
tb = ta + 4
tc = tb + 200

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

print(f'\nT = {48 / beta} GHz')
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
width = 0.03
plt.bar(expected_extracted_work, 1, width=width, color='C0', label=r'$-\Delta F$')
plt.bar(max_work, prob, width=width, color='C1', label=r'$W_{max}$')
plt.bar(min_work, 1-prob, width=width, color='C2', label=r'$W_{min}$')
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
plt.plot([s_low, s_low], [-0.1, 1.7], ls='--', c='black', alpha=0.5)
plt.plot([s_high, s_high], [-0.1, 1.7], ls='--', c='black', alpha=0.5)
plt.xlabel('s')
plt.ylabel('F (GHz)')
plt.title('free energy vs s')
plt.legend()

plt.figure('annealing schedule')
plt.plot([0, 1, ta], [1, s_low, s_low], c='C0', ls='--', label='preparation and measure')
plt.plot([ta, tb], [s_low, s_high], c='C0', marker='.', label='measured procedure')
plt.plot([tb, tc], [s_high, s_high], c='C0', marker='.', ls='-.', label='unmeasured procedure')
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

fig, axs = plt.subplots(2, 1)
ax = axs[0]
ax.plot([0, 1, ta, ta+1], [1, s_low, s_low, 1])
ax.scatter(ta, s_low, label='A', zorder=3)
ax.legend(loc='upper center')
ax.set_ylabel('s')
ax.set_title('measure A: $p(x_a)$')
ax.set_ylim([0.6, 1.05])

ax = axs[1]
ax.plot([0, 1, 5, 6], [1, s_low, s_high, 1])
ax.scatter(1, s_low, label='A', zorder=3)
ax.scatter(5, s_high, c='C1', label='B', zorder=3)
ax.legend(loc='upper center')
ax.set_title('measure B: $p(x_b|x_a)$')
ax.set_ylim([0.6, 1.05])

plt.tight_layout()

# RESULTS
QLp = 0.970083; dQLp = 0.00039
QHp = 0.999608; dQHp = 4.00e-05
Ap = -0.909559; dAp = 0.00062
Am = -0.910274; dAm = 0.00063
Bp = 0.8088681; dBp = 0.00089
Bm = -0.989549; dBm = 0.00022


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
width = 0.1
plt.bar(work_pp, p_pp, width=width, color='C1', label='pp')
plt.bar(work_mm, p_mm, width=width, color='C2', label='mm')
plt.bar(work_pm, p_pm, width=width, color='C3', label='pm')
plt.bar(work_mp, p_mp, width=width, color='C4', label='mp')

plt.xlabel('W')
plt.ylabel('p(W)')
plt.title('probability of work extraction distribution')
plt.legend()

plt.figure('work hist')
width = 0.04
plt.bar(max_work, prob, align='edge', width=width, label=r'expected $W_{max}$')
plt.bar(min_work, 1 - prob, align='edge', width=width, label=r'expected $W_{min}$')
plt.bar(expected_extracted_work, 1, width=width, label=r'expected $-\Delta F$')

plt.bar(work_pp, p_pp, align='edge', width=-width, label='pp')
plt.bar(work_mm, p_mm, align='edge', width=-width, label='mm')
plt.bar(work_pm, p_pm, width=width, label='pm')
plt.bar(work_mp, p_mp, width=width, label='mp')
actual_work = work_pp * p_pp + work_mm * p_mm + work_pm * p_pm + work_mp * p_mp
# plt.bar(actual_work, 1, align='edge', width=-width, label=r'expected $-\Delta F$')

plt.title('probability of work extraction distribution - theory vs experiment')
plt.xlabel('W')
plt.ylabel('p(W)')
plt.legend()

plt.show()
