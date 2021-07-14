import numpy as np
import time
from dwave.system.samplers import DWaveSampler

# Global variables and problem setting
s_low = 0.65
s_high = 0.67
h = 0.15

ta = 50 + 1
tb = ta + 1
tc = tb + 100

num_reads = 200

anneal_schedules = [
    [[0, 1], [1, s_low], [2, 1]],
    [[0, 1], [1, s_high], [2, 1]],
    [[0, 1], [1, s_low], [ta, s_low], [ta+1, 1]],
    [[0, 1], [1, s_low], [2, s_high], [3, 1]],
    [[0, 1], [1, s_low], [tc-tb+1, s_high], [tc-tb+2, 1]]
    ]
names = ['QL', 'QH', 'A', 'B', 'C']
spin_choice = [(1, ), (1, ), (1, -1), (1, -1), (1, -1)]

sampler = DWaveSampler(solver='DW_2000Q_6')
tic = time.time()
print('# measure spin <sigma_z> d<sigma_z>')

for anneal_schedule, name, spins in zip(anneal_schedules, names, spin_choice):
    for spin in spins:
        linear_offsets = {active_node: h for active_node in sampler.nodelist}
        initial_state = {active_node: spin for active_node in sampler.nodelist}

        E_fin = []
        sampleset = sampler.sample_ising(linear_offsets, {}, initial_state=initial_state,
                                         anneal_schedule=anneal_schedule, auto_scale=False, num_reads=num_reads,
                                         label=f'Measure {str(spin)}, {name}')
        for record in sampleset.record:
            E_fin += [record.energy] * record.num_occurrences
        num = len(linear_offsets)

        final_energy_arr = np.array(E_fin) / num / h
        mean_spin = np.mean(final_energy_arr)
        std_spin = np.std(final_energy_arr) / np.sqrt(num_reads)

        print(name, str(spin), mean_spin, std_spin)

toc = time.time()
print('# Elapsed time:', toc-tic, 's.')
