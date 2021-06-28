import numpy as np
import pickle
import time
from dwave.system.samplers import DWaveSampler
from settings import DATA_DIR

# Global variables and problem setting
anneal_param_min_list = [0.69]  # np.arange(0.61, 0.70, 0.01)
anneal_pause_lenght = list(range(0, 21, 2))
num_reads = 50
h0 = 0.95

sampler = DWaveSampler(solver='DW_2000Q_6')
h = {active_node: h0 for active_node in sampler.nodelist}
initial_state = {active_node: 1 for active_node in sampler.nodelist}

for anneal_param_min in anneal_param_min_list:
    # Thermalization at different times for a fixed anneal_param
    mean_spin = []
    std_spin = []

    tic = time.time()
    for pause in anneal_pause_lenght:
        E_fin = []
        schedule = [[0, 1], [1, anneal_param_min], [1 + pause, anneal_param_min], [2 + pause, 1]]
        if pause == 0:
            schedule = [[0, 1], [1, anneal_param_min], [2 + pause, 1]]

        sampleset = sampler.sample_ising(h, {}, initial_state=initial_state, anneal_schedule=schedule, auto_scale=False,
                                         num_reads=num_reads, label=f"s_t={anneal_param_min}, pause_lenght={pause}")

        for record in sampleset.record:
            E_fin += [record.energy] * record.num_occurrences

        final_energy_arr = np.array(E_fin) / len(h)
        mean_spin.append(np.mean(final_energy_arr))
        std_spin.append(np.std(final_energy_arr) / np.sqrt(num_reads))

    toc = time.time()

    print('\nThermalization at s =', anneal_param_min, 'completed.')
    print('Elapsed time:', toc-tic, 's.')

    with open(DATA_DIR / f'sbar_{round(anneal_param_min, 4)}_h{h0}.pkl', 'wb') as f:
        pickle.dump((mean_spin, std_spin), f)

    print('Saved data into ->', f.name)
