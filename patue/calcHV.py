import os
import sys 
sys.path.append(".")

from utils.utils import *

nobjs = 3
refpoint = 150.0
results = [[99.27526683357492, 95.87000618748448, 97.58779396002774], [99.90776123336408, 98.89335013818778, 92.54649769875796], [97.97074713400976, 109.35417647839377, 101.04911417943383]]
print("[Hypervolume]:", calcHypervolume([refpoint] * nobjs, results))
print("[Hypervolume 0,1]:", calcHypervolume([refpoint] * 2, np.array(results)[:, 0:2]))
print("[Hypervolume 0,2]:", calcHypervolume([refpoint] * 2, np.array(results)[:, [0,2]]))
print("[Hypervolume 1,2]:", calcHypervolume([refpoint] * 2, np.array(results)[:, 1:3]))
print("[MPI1]:", max(map(lambda x: 100.0 - x[0], results)))
print("[MPI2]:", max(map(lambda x: 100.0 - x[1], results)))
print("[MAI]:", max(map(lambda x: 100.0 - x[2], results)))
print("[MPPI]:", max(map(lambda x: 100.0 - x[0]*x[1]/100.0, results)))
print("[MPPAI]:", max(map(lambda x: 100.0 - x[0]*x[1]*x[2]/10000.0, results)))



