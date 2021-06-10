import numpy as np
import pickle
import time
from dwave.system.samplers import DWaveSampler
from settings import DATA_DIR

# Global variables and problem setting
num_reads = 50

sampler = DWaveSampler(solver='DW_2000Q_6')
h = {active_node: 0 for active_node in sampler.nodelist}
initial_state = {active_node: 1 for active_node in sampler.nodelist}

connections = [(n, n + 4) for n in sampler.nodelist if (n % 8 < 4) and (n, n + 4) in sampler.edgelist]
J = {active_coupler: 1 for active_coupler in connections}

# Thermalization at different times for a fixed anneal_param
anneal_param = 0.8  # the lowest value allowed and the closest thing to "no driving" we can achieve

mean_E = []

tic = time.time()
for anneal_lenght in np.linspace(1, 100, num=10):
    E_fin = 0
    schedule = [[0, 1], [anneal_lenght / 2, anneal_param], [anneal_lenght, 1]]
    sampleset = sampler.sample_ising(h, J, initial_state=initial_state, anneal_schedule=schedule, auto_scale=False, num_reads=num_reads)
    for record in sampleset.record:
        E_fin += record.energy * record.num_occurrences
    mean_E.append(E_fin / len(J) / num_reads)
toc = time.time()

print('Thermalization at s =', anneal_param, 'completed.\n')
print('Elapsed time:', toc-tic, 's.\n')

with open(DATA_DIR / 'experimental_results_B2.pkl', 'wb') as f:
    pickle.dump(mean_E, f)

print('Saved data into ->', f.name)
