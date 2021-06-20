import numpy as np
import pandas as pd
import scipy.interpolate
import scipy.stats
from matplotlib import pyplot as plt
from settings import ANNEALING_SCHEDULE_XLS

s_init = 1
s_a = 0.68
s_b = 0.70
s_fin = s_a
h = 0.2

ta = 20
tb = 21
tc = 60
td = 100

# annealing functions
df = pd.read_excel(ANNEALING_SCHEDULE_XLS, sheet_name=1)
A = scipy.interpolate.CubicSpline(df['s'], df['A(s) (GHz)'])
B = scipy.interpolate.CubicSpline(df['s'], h * df['B(s) (GHz)'])
beta = 48 / 13.5  # 1 / GHz


def free_energy(E):
    return - np.log(np.exp(- beta * E) + 1) / beta


minus_delta_f = free_energy(B(s_init)) - free_energy(B(s_fin))
max_work = minus_delta_f + free_energy(B(s_b)) - free_energy(B(s_a))
min_work = max_work + B(s_a) - B(s_b)
prob = (np.exp(beta * minus_delta_f) - np.exp(beta * min_work)) / (np.exp(beta * max_work) - np.exp(beta * min_work))

print(f'-∆F = {minus_delta_f:.4e} GHz')
print(f'W_max = {max_work:.4e} GHz')
print(f'W_min = {min_work:.4f} GHz')

print(f'\nT = {1/beta} GHz')
if minus_delta_f != 0:
    print(f'W_max - (-∆F) = {max_work - minus_delta_f:.4e} GHz.')
    print(f'(W_max - (-∆F)) / (-∆F) = {100 * (max_work - minus_delta_f) / minus_delta_f:.2f}%')
print('p =', prob, '\n')

for x in []:
    print(f'E(s={x}) = {B(x):.2f} GHz')
    print(f'F(s={x}) = {free_energy(B(x)):.2e} Ghz')

x = np.linspace(0, 1, 1000)

plt.figure('annealing functions')
plt.plot(x, A(x), label='A(s) GHz')
plt.plot(x, B(x), label=f'{h} * B(s) GHz')
plt.plot([0, 1], [1 / beta] * 2, ls='--', label='E=kT')
plt.plot([s_a, s_a], [0, 10], ls='--', c='black', alpha=0.5)
plt.plot([s_b, s_b], [0, 10], ls='--', c='black', alpha=0.5)
plt.title('anneling functions')
plt.xlabel('s')
plt.ylabel('Energy (GHZ)')
plt.tight_layout()
plt.legend()

plt.figure('rapporti')
plt.yscale('log')
plt.plot(x, A(x) / B(x), label='A(s) / B(s)')
plt.plot(x, 1 / (B(x) * beta), label=f'kT / {h} * B(s)')
plt.plot([0, 1], [1, 1], ls='--', label='equal energy')
plt.plot([s_a, s_a], [5e-8, 5e2], ls='--', c='black', alpha=0.5)
plt.plot([s_b, s_b], [5e-8, 5e2], ls='--', c='black', alpha=0.5)
plt.xlabel('s')
plt.tight_layout()
plt.legend()

plt.figure('work')
# plt.yscale('log')
plt.ylim((0, 1.05))
plt.plot([max_work, max_work], [0, prob], c='C1', label=r'$W_{max}$')
plt.plot([min_work, min_work], [0, 1-prob], c='C2', label=r'$W_{min}$')
plt.plot([minus_delta_f, minus_delta_f], [0, 1], c='C0', label=r'$-\Delta F$')  #, ls='--')
plt.xlabel('W')
plt.ylabel('p(W)')
plt.title('probability of work extraction distribution')
plt.legend()

plt.figure('free energy vs energy')
energy = np.linspace(0.5, 10, 1000)
plt.xscale('log')
plt.plot(energy, free_energy(energy))
plt.plot([B(s_a), B(s_a)], [-0.044, 0], ls='--', c='black', alpha=0.5)
plt.plot([B(s_b), B(s_b)], [-0.044, 0], ls='--', c='black', alpha=0.5)
plt.xlabel('energy: E (GHz)')
plt.ylabel('Free energy: F (GHz)')
plt.title('free energy vs energy')

plt.figure('free energy vs s')
x = np.linspace(0.5, 1, 1000)
plt.plot(x, free_energy(B(x)), label='F(E): free energy')
plt.plot(x, B(x), label='E: energy')
plt.plot([s_a, s_a], [-0.044, 2], ls='--', c='black', alpha=0.5)
plt.plot([s_b, s_b], [-0.044, 2], ls='--', c='black', alpha=0.5)
plt.xlabel('s')
plt.ylabel('F (GHz)')
plt.title('free energy vs s')
plt.legend()

plt.figure('annealing schedule')
plt.plot([0, ta], [1, s_a], c='C0', ls='--')
plt.plot([td, td+1], [s_fin, 1], c='C0', ls='--')
plt.plot([ta, tb, tc, td], [s_a, s_b, s_b, s_fin], c='C0', marker='.')
plt.text(ta + 0.7, s_a - 0.005, 'a')
plt.text(tb - 0.7, s_b + 0.005, 'b')
plt.text(tc - 0.7, s_b + 0.005, 'c')
plt.text(td + 0.7, s_fin - 0.005, 'd')
plt.xlabel('t (µs)')
plt.ylabel('s')
plt.title('annealing schedule')

plt.figure('invasive measures')
plt.plot([0, 1, 2], [1, s_a, 1])
plt.title('measure quench')
plt.xlabel('t (µs)')
plt.ylabel('s')

fig, axs = plt.subplots(2, 2)
ax = axs[0, 0]
ax.plot([0, ta, ta+1], [1, s_a, 1])
ax.scatter(ta, s_a)
ax.set_ylabel('s')
ax.set_title('measure A: $p(x_a)$')

ax = axs[0, 1]
ax.plot([0, 1, 2, 3], [1, s_a, s_b, 1])
ax.scatter(1, s_a)
ax.scatter(2, s_b, c='C1')
ax.set_title('measure B: $p(x_b|x_a)$')

ax = axs[1, 0]
ax.plot([0, 1, tc - tb + 1, tc - tb + 2], [1, s_b, s_b, 1])
ax.scatter(1, s_b, c='C1')
ax.scatter(tc - tb + 1, s_b, c='C2')
ax.set_xlabel('t (µs)')
ax.set_ylabel('s')
ax.set_title('measure C: $p(x_c|x_b)$')

ax = axs[1, 1]
ax.plot([0, 1, td - tc + 1, td - tc + 2], [1, s_b, s_fin, 1])
ax.scatter(1, s_b, c='C2')
ax.scatter(td - tc + 1, s_fin, c='C3')
ax.set_xlabel('t (µs)')
ax.set_title('measure D: $p(x_d|x_c)$')

plt.tight_layout()
plt.show()
