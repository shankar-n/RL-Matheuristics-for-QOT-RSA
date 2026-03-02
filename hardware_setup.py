import time
import torch
from pyscipopt import Model, quicksum

# ==========================================
# 0. HARDWARE SETUP & CALIBRATION
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Executing Deep Learning on: {device}")

def get_machine_calibration_factor():
    start = time.perf_counter()
    m = Model("calibration")
    m.hideOutput()
    # 100-variable LP provides a stable, measurable CPU load to prevent OS variance
    vars = [m.addVar(vtype="I", lb=0, ub=10) for _ in range(100)]
    m.setObjective(quicksum(v for v in vars), "maximize")
    for i in range(99):
        m.addCons(vars[i] + 2*vars[i+1] <= 20)
    m.optimize()
    duration = max(time.perf_counter() - start, 1e-5)
    reference_time = 0.05 
    return duration / reference_time

MACHINE_SPEED_FACTOR = get_machine_calibration_factor()
print(f"Machine Calibration Factor: {MACHINE_SPEED_FACTOR:.3f}")