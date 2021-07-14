import numpy as np
import pickle
import time
from dwave.system.samplers import DWaveSampler
from settings import DATA_DIR

# Global variables and problem setting
s_pause = np.linspace(0, 1, 51)
anneal_pause_lenght = np.linspace(0, 80, 5)
num_reads = 100
h0 = 0.3

sampler = DWaveSampler(solver='DW_2000Q_6')
h = {active_node: h0 for active_node in sampler.nodelist}

for sbar in s_pause:
    # Thermalization at different times for a fixed anneal_param
    mean_spin = []
    std_spin = []

    tic = time.time()
    for pause in anneal_pause_lenght:
        E_fin = []
        schedule = [[0, 0], [1, sbar], [1 + pause, sbar], [2 + pause, 1]]
        if pause == 0:
            schedule = [[0, 0], [1, sbar], [2 + pause, 1]]

        sampleset = sampler.sample_ising(h, {}, anneal_schedule=schedule, auto_scale=False,
                                         num_reads=num_reads, label=f"s_t={sbar}, pause_lenght={pause}")

        for record in sampleset.record:
            E_fin += [record.energy] * record.num_occurrences

        final_energy_arr = np.array(E_fin) / len(h)
        mean_spin.append(np.mean(final_energy_arr))
        std_spin.append(np.std(final_energy_arr) / np.sqrt(num_reads))

    toc = time.time()

    print('\nThermalization at s_pause =', sbar, 'completed.')
    print('Elapsed time:', toc-tic, 's.')

    with open(DATA_DIR / f'6_temp_sbar_{sbar:.2f}_h{h0}.pkl', 'wb') as f:
        pickle.dump((mean_spin, std_spin), f)

    print('Saved data into ->', f.name)
