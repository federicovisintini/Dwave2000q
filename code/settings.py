from pathlib import Path
import numpy as np

MAIN_DIR = Path.cwd().parent
DATA_DIR = MAIN_DIR / 'data'
ANNEALING_SCHEDULE_XLS = MAIN_DIR / '09-1216A-A_DW_2000Q_6_annealing_schedule.xls'
S_LOW = 0.68
S_HIGH = 0.71
TA = 30
TC = 111
TD = 190

STS = [0.60, 0.65, 0.70, 0.75]  # 0.55
HS_LIST = [
    # [np.nan, 0.17, 0.323, 0.46, 0.718, 0.97, np.nan],
    [0.076, 0.20, 0.31, 0.417, 0.63, 0.84, np.nan],
    [0.088, 0.18, 0.273, 0.366, 0.55, 0.73, 0.92],
    [0.080, 0.16, 0.24, 0.32, 0.48, 0.64, 0.80],
    [0.071, 0.142, 0.212, 0.283, 0.425, 0.565, 0.71]
]