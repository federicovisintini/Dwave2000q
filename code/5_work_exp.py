import numpy as np
import time
from dwave.system.samplers import DWaveSampler
from settings import *

# Global variables and problem setting
h = 0.15
num_reads = 200
anneal_schedules = [
    [[0, 1], [1, S_LOW], [2, 1]],
    [[0, 1], [1, S_HIGH], [2, 1]],
    [[0, 1], [1, S_LOW], [TA, S_LOW], [TA+1, 1]],
    [[0, 1], [1, S_LOW], [2, S_HIGH], [3, 1]],
    [[0, 1], [1, S_HIGH], [TC-TA, S_HIGH], [TC-TA+1, 1]],
    [[0, 1], [1, S_HIGH], [TD-TC, S_LOW], [TD-TC+1, 1]]
    ]

names = ['QL', 'QH', 'A', 'B', 'C', 'D']


sampler = DWaveSampler(solver='DW_2000Q_6')
tic = time.time()
print('# measure spin <sigma_z> d<sigma_z>')

for anneal_schedule, name in zip(anneal_schedules, names):
    for spin in [1, -1]:
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
