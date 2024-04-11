import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ARDRegression, LinearRegression

if __name__ == "__main__":
    candidates = []
    names = []
    with open("historyMORBO.txt", "r") as fin: 
        infos = []
        results = []
        lines = fin.readlines()
        for idx in range(0, len(lines), 2): 
            if idx >= len(lines) - 1: 
                break
            info = lines[idx].strip()
            result = lines[idx+1].strip()
            if len(names) == 0: 
                names = list(eval(info).keys())
            info = list(eval(info).values())
            result = list(eval(result))
            while len(candidates) < len(info): 
                candidates.append(set())
            for jdx in range(len(info)): 
                candidates[jdx].add(info[jdx])
            infos.append(info)
            results.append(result)
    values = []
    for items in candidates: 
        tmp = {}
        for elem in items: 
            tmp[elem] = len(tmp)
        values.append(tmp)
    for idx in range(len(infos)): 
        for jdx in range(len(infos[idx])): 
            key = infos[idx][jdx]
            number = values[jdx][key]
            infos[idx][jdx] = number
    infos = np.array(infos)
    results = np.array(results)
    clf = ARDRegression(compute_score=True)
    clf.fit(infos, results[:, 0])
    coeff = np.array(clf.coef_)
    std = 1.0 / np.sqrt(np.array(clf.lambda_))
    for idx in range(1, 100): 
        clf = ARDRegression(compute_score=True)
        clf.fit(infos, results[:, 0])
        coeff += np.array(clf.coef_)
        std += 1.0 / np.sqrt(np.array(clf.lambda_))
    coeff /= 100.0
    std /= 100.0

    print(coeff)
    print(std)

    # indices = np.arange(0, len(coeff))
    # plt.subplot(1, 2, 1)
    # plt.bar(indices, coeff)
    # plt.subplot(1, 2, 2)
    # plt.bar(indices, std)
    # plt.show()

    name2coeff = dict(zip(names, list(coeff)))
    name2std = dict(zip(names, list(std)))
    sortedNames = sorted(names, key=lambda x: -name2std[x])
    for name in sortedNames: 
        print(f"{name}: std: {name2std[name]}, coeff: {name2coeff[name]}")

    examples = ["syn_generic_effort", "syn_map_effort", "global_timing_effort", "global_cong_effort", 
                "use_scan_seqs_for_non_dft", "wire_opt", "litho_driven", 
                "optimize_seq_x_to", "pre_place_opt", "detail_irdrop_aware", "ultra_global_mapping", ]
    plt.yscale("log")
    for idx in range(len(examples)): 
        plt.bar(idx, name2std[examples[idx]])
    plt.legend(examples)
    import tikzplotlib
    tikzplotlib.save("barImp.tex")
    plt.show()

