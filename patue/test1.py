import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ARDRegression, LinearRegression


names = ["syn_generic_effort", "syn_map_effort", "global_timing_effort", "global_cong_effort", 
         "use_scan_seqs_for_non_dft", "wire_opt", "litho_driven", 
         "optimize_seq_x_to", "pre_place_opt", "detail_irdrop_aware", "ultra_global_mapping", ]
values = [21.0612528329719, 20.9670060268702, 35.1017383245309, 0.126492099682508, 
          0.146137024661059, 0.386812611522991, 0.17688105238215, 
          17.478839345396, 0.190511166829219, 0.110986114698506, 0.151955862673766]
logs = []
for value in values: 
    logs.append(math.log10(value) + 1.0)
print(logs)
exit(1)

plt.figure(figsize=(5, 4))
plt.yscale("log")
for idx in range(len(names)): 
    plt.bar(idx, values[idx])
plt.legend(names, bbox_to_anchor=(1, 1), loc=2, borderaxespad=0)
plt.show()


