import pickle
import numpy as np
from matplotlib import pyplot as plt


def state_population(_linear_offsets, _quadratic_coupling, _sampleset):
    # calcoliamo la percentuale di popolamento gli stati lungo lungo z
    _state = np.zeros(4)

    for a, b in _quadratic_coupling:
        variables = np.array(_sampleset.variables)

        for sample in _sampleset.record.sample:
            index_a = np.where(variables == a)[0][0]
            index_b = np.where(variables == b)[0][0]
            x = sample[index_a]
            y = sample[index_b]

            if np.sign(_linear_offsets[a]) < 0:
                x = -x
                y = -y

            if x == -1 and y == -1:
                _state[0] += 1
            elif x == 1 and y == 1:
                _state[1] += 1
            elif x == -1 and y == 1:
                _state[2] += 1
            elif x == 1 and y == -1:
                _state[3] += 1
            else:
                raise ValueError

    _err = 1 / np.sqrt(4 * len(_quadratic_coupling)
                       * _sampleset.record.num_occurrences.sum())

    return _state / _state.sum(), _err


def load_2qubits(N):
    # load 2 qubit data
    _spin_up2 = np.empty(N)
    _dspin_up2 = np.empty(N)
    _biases2 = np.linspace(-0.3, 0.3, N)

    for i in range(N):
        # load data from file
        linear_offsets, quadratic_couplings, sampleset = pickle.load(
            open(f"data/2_qubits_boltzmann{i}.pickle", "rb"))

        # compute state population and error
        state, err = state_population(linear_offsets, quadratic_couplings, sampleset)
        if state[2] != 0 or state[3] != 0:
            print(f'h: {_biases2[i]}, state pop: {state}')

        # append result when both qubits are spin 'up'
        if _biases2[i] > 0:
            _spin_up2[i] = state[0]
        else:
            _spin_up2[i] = state[1]
        _dspin_up2[i] = err
    return _biases2, _spin_up2, _dspin_up2


# load data
biases1, spin_up1, dspin_up1 = pickle.load(open("data/results.pickle", "rb"))
biases2, spin_up2, dspin_up2 = load_2qubits(20)
pickle.dump((biases2, spin_up2, dspin_up2), open("data/results2.pickle", "wb"))

# grafico popolazione ground state vs h - 2 qubits
plt.figure()
plt.errorbar(biases1, spin_up1, dspin_up1, marker='.', linestyle='', label='1 qubit')
plt.errorbar(biases2, spin_up2, dspin_up2, marker='.', linestyle='', label='2 qubits')
plt.title('two qubits temperature')
plt.xlabel('h')
plt.ylabel(r'$P_\uparrow$')
plt.legend()
plt.show()
