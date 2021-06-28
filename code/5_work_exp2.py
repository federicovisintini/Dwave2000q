import numpy as np
import time
import pickle
from dwave.system.samplers import DWaveSampler
from settings import *

# Global variables and problem setting
h = 0.15
num_reads = 50

measure_times = list(range(1, TD+1, 5))
anneal = [[0, 1], [1, S_LOW], [TA, S_LOW], [TA+1, S_HIGH], [TC, S_HIGH], [TD, S_LOW], [TD+1, 1]]
anneal_schedules = []

for final_time in measure_times:
    measure_anneal_schedule = [x for x in anneal if x[0] <= final_time]
    previous_point = measure_anneal_schedule[-1]
    successive_point = [x for x in anneal if x[0] > final_time][0]
    middle_s = previous_point[1] + (final_time - previous_point[0]) / (successive_point[0] - previous_point[0]) * (successive_point[1] - previous_point[1])
    if [final_time, middle_s] != measure_anneal_schedule[-1]:
        measure_anneal_schedule.append([final_time, round(middle_s, 5)])
    measure_anneal_schedule.append([final_time + 1, 1])

    anneal_schedules.append(measure_anneal_schedule)

sampler = DWaveSampler(solver='DW_2000Q_6')
tic = time.time()


for anneal_schedule in anneal_schedules:
    linear_offsets = {active_node: h for active_node in sampler.nodelist}
    initial_state = {active_node: -1 for active_node in sampler.nodelist}

    E_fin = []
    sampleset = sampler.sample_ising(linear_offsets, {}, initial_state=initial_state,
                                     anneal_schedule=anneal_schedule, auto_scale=False, num_reads=num_reads,
                                     label=f'T_measure =' + str(anneal_schedule[-1][0]))
    for record in sampleset.record:
        E_fin += [record.energy] * record.num_occurrences
    num = len(linear_offsets)

    final_energy_arr = np.array(E_fin) / num / h
    mean_spin = np.mean(final_energy_arr)
    std_spin = np.std(final_energy_arr) / np.sqrt(num_reads)

    with open(DATA_DIR / f'work_extraction_naive0_tf{anneal_schedule[-1][0]}.pkl', 'wb') as f:
        pickle.dump((anneal_schedule, mean_spin, std_spin), f)

toc = time.time()
print('# Elapsed time:', toc-tic, 's.')
