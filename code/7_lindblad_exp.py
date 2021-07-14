import numpy as np
import json
from dwave.system.samplers import DWaveSampler
from settings import DATA_DIR, STS, HS_LIST

# Global variables and problem setting
anneal_pause_lenght = range(0, 11)
num_reads = 120

sampler = DWaveSampler(solver='DW_2000Q_6')
mean_spin_dict = {}
std_spin_dict = {}
for st, hs in zip(STS, HS_LIST):
    for h0 in hs:
        initial_state = {active_node: np.random.choice([-1, 1]) for active_node in sampler.nodelist}
        h = {active_node: initial_state[active_node] * h0 for active_node in sampler.nodelist}

        mean_spin = []
        std_spin = []
        for pause in anneal_pause_lenght:
            E_fin = []
            schedule = [[0, 1], [1, st], [1 + pause, st], [2 + pause, 1]]
            if pause == 0:
                schedule = [[0, 1], [1, st], [2 + pause, 1]]

            sampleset = sampler.sample_ising(h, {}, initial_state=initial_state, anneal_schedule=schedule,
                                             auto_scale=False, num_reads=num_reads,
                                             label=f"s_t={st}, pause_lenght={pause}", reinitialize_state=True)

            for record in sampleset.record:
                E_fin += [record.energy] * record.num_occurrences

            final_energy_arr = np.array(E_fin) / len(h) / h0
            mean_spin.append(np.mean(final_energy_arr))
            std_spin.append(np.std(final_energy_arr) / np.sqrt(len(final_energy_arr)))

        key_list = [str(st), str(h0)]
        key = ":".join(key_list)
        mean_spin_dict[key] = mean_spin
        std_spin_dict[key] = std_spin

with open(DATA_DIR / '7_experiments_mean_spin.json', 'w') as f:
    json.dump(mean_spin_dict, f, indent=4)

with open(DATA_DIR / '7_experiments_std_spin.json', 'w') as g:
    json.dump(std_spin_dict, g, indent=4)

print('Saved data into ->', f.name, g.name)
