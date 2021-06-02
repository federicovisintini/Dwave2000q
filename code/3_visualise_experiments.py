import numpy as np
import json
import matplotlib.pyplot as plt
from settings import DATA_DIR


with open(DATA_DIR / "multiple_experiments.json", "r") as read_file:
    data = json.load(read_file)

plt.figure()

# 1 qubit
data1 = [obj for obj in data if obj["num_qubits"] == 1]

x1 = np.array([obj["linear_offsets"][0] for obj in data1])
state_p = np.array([obj["states"]["p"] for obj in data1])
state_m = np.array([obj["states"]["m"] for obj in data1])

up1 = state_m / (state_m + state_p)
std1 = np.sqrt(state_p * state_m / (state_m + state_p)**3)
plt.errorbar(x1, up1, std1, label='1 qubit')


# 2 qubits
data2 = [obj for obj in data if obj["num_qubits"] == 2]
hB_values = np.unique([obj["linear_offsets"][1] for obj in data2])
J_values = np.unique([obj["quadratic_couplings"]["01"] for obj in data2])

for hB in hB_values:
    for J in J_values:
        x = np.array([obj["linear_offsets"][0] for obj in data2
                      if obj["quadratic_couplings"]["01"] == J and obj["linear_offsets"][1] == hB])
        state_pp = np.array([obj["states"]["pp"] for obj in data2
                             if obj["quadratic_couplings"]["01"] == J and obj["linear_offsets"][1] == hB])
        state_pm = np.array([obj["states"]["pm"] for obj in data2
                             if obj["quadratic_couplings"]["01"] == J and obj["linear_offsets"][1] == hB])
        state_mp = np.array([obj["states"]["mp"] for obj in data2
                             if obj["quadratic_couplings"]["01"] == J and obj["linear_offsets"][1] == hB])
        state_mm = np.array([obj["states"]["mm"] for obj in data2
                             if obj["quadratic_couplings"]["01"] == J and obj["linear_offsets"][1] == hB])


        num_samplings = state_pp + state_pm + state_mp + state_mm
        ground = (state_mp + state_mm) / num_samplings
        std = np.sqrt(state_mm * (num_samplings - state_mm) / num_samplings ** 3)
        plt.errorbar(x, ground, std, marker = '.', label=f'hB={hB}, J={J}')

plt.xlabel('hA')
plt.ylabel('spin A is up')
plt.title('multiple experiments')
plt.legend()
plt.show()