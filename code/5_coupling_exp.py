import numpy as np
import pickle
import time
from dwave.system.samplers import DWaveSampler
from settings import DATA_DIR

# Global variables and problem setting
anneal_param_min_list = np.arange(0.75, 0.78, 0.01)
anneal_lenght = np.linspace(5, 50, num=10)
num_reads = 50

sampler = DWaveSampler(solver='DW_2000Q_6')
h = {active_node: 1 for active_node in sampler.nodelist}
initial_state = {active_node: 1 for active_node in sampler.nodelist}

for anneal_param_min in anneal_param_min_list:
    # Thermalization at different times for a fixed anneal_param
    mean_E = []

    tic = time.time()
    for tf in anneal_lenght:
        E_fin = 0
        schedule = [[0, 1], [tf / 2, anneal_param_min], [tf, 1]]
        sampleset = sampler.sample_ising(h, {}, initial_state=initial_state, anneal_schedule=schedule, auto_scale=False,
                                         num_reads=num_reads, label=f"s_t={anneal_param_min}, tf={tf}")
        for record in sampleset.record:
            E_fin += record.energy * record.num_occurrences
        num = len(h) * num_reads

        mean_E.append(E_fin / num)
    toc = time.time()

    print('\nThermalization at s =', anneal_param_min, 'completed.')
    print('Elapsed time:', toc-tic, 's.')

    with open(DATA_DIR / f'coupling_st{anneal_param_min}.pkl', 'wb') as f:
        pickle.dump(mean_E, f)

    print('Saved data into ->', f.name)
