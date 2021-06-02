import numpy as np
import random
import math
import json
from settings import DATA_DIR
from dwave.system import DWaveSampler


def energy_state(qubits: list, biases: list, couplings):
    """
    Computes the energy of a Ising problem; given spins, biases and couplings.

    Parameters
    ----------
    qubits: list
        list of 1 or -1, orientation of the i-th spin along z-axis
    biases: list
        list where i-th element is the bias acting on the i-th qubit
    couplings: dict
        keys are the couple of qubits connected (eg. (2,4))
        values are the strenght of the coupling

    Returns
    -------
    energy: float
        the energy of the configuration (negative is energetically favored)
    """

    energy = 0
    for q in range(len(biases)):
        energy -= biases[q] * qubits[q]
    for q1, q2 in couplings:
        energy -= couplings[(q1, q2)] * qubits[q1] * qubits[q2]
    return energy


def single_qubit_sample(num_reads, annealing_t, offsets):
    sampler = DWaveSampler(solver='DW_2000Q_6')
    list_json = []

    for h in offsets:
        linear_offsets = {active_node: h * random.choice([-1, 1]) for active_node in sampler.nodelist}
        sampleset = sampler.sample_ising(linear_offsets, {}, num_reads=num_reads,
                                         annealing_time=annealing_t, auto_scale=False, label=f'1, h={h:.3f}')

        energy_p = energy_state([1], [h], {})
        energy_m = energy_state([-1], [h], {})

        # cycle on every problem to see energetic level pop
        pop_p = 0
        pop_m = 0
        for record in sampleset.record:
            result_qubits = record[0]
            num_occurrences = record[2]
            for j, node in enumerate(linear_offsets):
                # evaluate problem experimental solution
                actual_energy = energy_state([result_qubits[j]], [linear_offsets[node]], {})
                if math.isclose(actual_energy, energy_p):
                    pop_p += num_occurrences
                elif math.isclose(actual_energy, energy_m):
                    pop_m += num_occurrences
                else:
                    print('Qualcosa non va')

        states = {
            'p': int(pop_p),
            'm': int(pop_m)
        }

        obj = {
            "num_qubits": 1,
            "annealing_time": annealing_t,
            "linear_offsets": [h],
            "quadratic_couplings": {},
            "states": states,
        }

        list_json.append(obj)

    return list_json


def double_qubits_sample(num_reads, annealing_t, offsets, hB, J):
    sampler = DWaveSampler(solver='DW_2000Q_6')
    list_json = []

    connections = [(n, n + 4) for n in sampler.nodelist if (n % 8 < 4) and (n, n + 4) in sampler.edgelist]
    quadratic_coupling = {active_coupler: J for active_coupler in connections}

    active_nodelist = []
    for q1, q2 in quadratic_coupling:
        active_nodelist.append(q1)
        active_nodelist.append(q2)

    sorted_nodelist = sorted(active_nodelist)

    for h in offsets:
        name = f"2 qubits, J={J}, hB ={hB}, time={annealing_t}"
        linear_offsets = {}
        for a, b in connections:
            randsign = random.choice([-1, 1])
            linear_offsets[a] = h * randsign
            linear_offsets[b] = hB * randsign

        sampleset = sampler.sample_ising(linear_offsets, quadratic_coupling, num_reads=num_reads,
                                         annealing_time=annealing_t, auto_scale=False,
                                         label=f'2, hA={h:.3f}, hB={hB}, J={J}')

        energy_pp = energy_state([1, 1], [h, hB], {(0, 1): J})
        energy_pm = energy_state([1, -1], [h, hB], {(0, 1): J})
        energy_mp = energy_state([-1, 1], [h, hB], {(0, 1): J})
        energy_mm = energy_state([-1, -1], [h, hB], {(0, 1): J})

        # ground state pop
        pop_pp = 0
        pop_pm = 0
        pop_mp = 0
        pop_mm = 0
        for record in sampleset.record:
            result_qubits = record[0]
            num_occurrences = record[2]

            for q1, q2 in quadratic_coupling:
                # evaluate problem experimental solution
                pos_q1 = sorted_nodelist.index(q1)
                pos_q2 = sorted_nodelist.index(q2)
                qubits_tmp = [result_qubits[pos_q1], result_qubits[pos_q2]]
                biases_tmp = [linear_offsets[q1], linear_offsets[q2]]
                coupling_tmp = {(0, 1): quadratic_coupling[(q1, q2)]}

                actual_energy = energy_state(qubits_tmp, biases_tmp, coupling_tmp)

                if math.isclose(actual_energy, energy_pp):
                    pop_pp += num_occurrences
                elif math.isclose(actual_energy, energy_pm):
                    pop_pm += num_occurrences
                elif math.isclose(actual_energy, energy_mp):
                    pop_mp += num_occurrences
                elif math.isclose(actual_energy, energy_mm):
                    pop_mm += num_occurrences
                else:
                    print('Qualcosa non va')

        states = {
            'pp': int(pop_pp),
            'pm': int(pop_pm),
            'mp': int(pop_mp),
            'mm': int(pop_mm)
        }

        obj = {
            "num_qubits": 2,
            "annealing_time": annealing_t,
            "linear_offsets": [h, hB],
            "quadratic_couplings": {'01': J},
            "states": states,
        }

        list_json.append(obj)

    return list_json


if __name__ == '__main__':
    num_different_h = 30
    num_qubit_reads = 50
    annealing_time = 5  # mu s
    x = np.linspace(-1, 1, num_different_h)
    hA = (x ** 3 + x) / 5

    # possible values of hB and J
    hB_values = [0, 0.1]
    J_values = [1, 0.1, 0.01]

    data = single_qubit_sample(num_qubit_reads, annealing_time, hA)

    for hB_2 in hB_values:
        for J_2 in J_values:
            data += double_qubits_sample(num_qubit_reads, annealing_time, hA, hB_2, J_2)

    with open(DATA_DIR / 'multiple_experiments.json', 'w') as fp:
        json.dump(data, fp, indent=4)
