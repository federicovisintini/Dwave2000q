import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.interpolate
from settings import ANNEALING_SCHEDULE_XLS

init = 1
sa = 0.8
sb = 0.7
fin = 0.6

# annealing functions
df = pd.read_excel(ANNEALING_SCHEDULE_XLS, sheet_name=1)
A = scipy.interpolate.CubicSpline(df['s'], df['A(s) (GHz)'])
B = scipy.interpolate.CubicSpline(df['s'], df['B(s) (GHz)'])
nominal_temp_ghz = 13.5 / 47.9924341590788

# plot annealing functions vs annealing parameter and plotting nominal temperature for comparison
plt.figure("anneling functions work extraction", figsize=(8, 6))
plt.plot(df['s'], df['A(s) (GHz)'], label='A(s)')
plt.plot(df['s'], df['B(s) (GHz)'], label='B(s)')

ydeltapos = -0.5
plt.scatter(init, B(init), c='C1')
plt.annotate('init', (init, B(init) + ydeltapos))
plt.scatter(sa, B(sa), c='C1')
plt.annotate('a', (sa, B(sa) + ydeltapos))
plt.scatter(sb, B(sb), c='C1')
plt.annotate('b', (sb, B(sb) + ydeltapos))
plt.scatter(fin, B(fin), c='C1')
plt.annotate('final', (fin, B(fin) + ydeltapos))

# plt.plot(s_bar * np.ones(50), np.linspace(0, 12), linestyle='--', color='black')
plt.plot(np.linspace(0, 1), nominal_temp_ghz * np.ones(50), linestyle='--', label='E = $k_B$T')
plt.title('Annealing functions')
plt.xlabel('annealing paramenter: s')
plt.ylabel('Energy (GHz)')
plt.legend()


ta = 10
tb = 11
tpause = 40
tf = 50
tend = 100


# annealing schedule
def t_to_s(t):
    if t <= ta:
        return 1 - t / ta * (1 - sa)
    elif ta < t <= tb:
        return sa + (sb - sa) * (t - ta) / (tb - ta)
    elif tb < t <= tpause:
        return sb
    elif tpause < t <= tf:
        return sb + (fin - sb) * (t - tpause) / (tf - tpause)
    elif tf < t <= tend:
        return fin + (1 - fin) * (t - tf) / (tend - tf)


t_anneal = np.linspace(0, 100, 1000)
s_anneal = np.empty_like(t_anneal)
for i, t_ in enumerate(t_anneal):
    s_anneal[i] = t_to_s(t_)

plt.figure("annealing schedule work extraction")
plt.plot(t_anneal[:500], s_anneal[:500], c='C0')
plt.plot(t_anneal[500:], s_anneal[500:], ls='--', c='C0')

plt.annotate('init', (1, t_to_s(0)))
plt.annotate('a', (ta + 1, t_to_s(ta)))
plt.annotate('b', (tb, t_to_s(tb)-0.05))
plt.annotate('end \npause', (tpause - 4, t_to_s(tpause) + 0.02))
plt.annotate('final', (tf, t_to_s(tf)-0.05))

plt.scatter([0, ta, tb, tpause, tf, tend], [1, sa, sb, sb, fin, 1], c='C0')
plt.xlabel("annealing time: t (µs)")
plt.ylabel("annealing parameter: s")
plt.title(r"annealing schedule")
plt.ylim((0, 1.1))


plt.figure("A, B as function of time work extraction")
plt.plot(t_anneal[:500], A(s_anneal[:500]), c='C0', label='A(t)')
plt.plot(t_anneal[500:], A(s_anneal[500:]), ls='--', c='C0')
plt.plot(t_anneal[:500], B(s_anneal[:500]), c='C1', label='B(t)')
plt.plot(t_anneal[500:], B(s_anneal[500:]), ls='--', c='C1')

plt.scatter([0, ta, tb, tpause, tf, tend], A([1, sa, sb, sb, fin, 1]), c='C0')
plt.scatter([0, ta, tb, tpause, tf, tend], B([1, sa, sb, sb, fin, 1]), c='C1')

plt.annotate('init', (1, B(t_to_s(0))))
plt.annotate('a', (ta + 1, B(t_to_s(ta))))
plt.annotate('b', (tb, B(t_to_s(tb))-0.5))
plt.annotate('end \npause', (tpause - 4, B(t_to_s(tpause)) + 0.1))
plt.annotate('final', (tf, B(t_to_s(tf))-0.5))

plt.title('annealing functions')
plt.xlabel("annealing time: t (µs)")
plt.ylabel('Energy (GHz)')
plt.legend()
plt.show()
