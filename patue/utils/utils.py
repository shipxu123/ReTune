import os
import sys 
sys.path.append(".")

import numpy as np 
import subprocess as sp

import torch
import botorch
from botorch.utils.multi_objective.hypervolume import Hypervolume

def calcHypervolume(refpoint, pareto): 
    model = Hypervolume(-torch.tensor(refpoint))
    return model.compute(-torch.tensor(pareto))

def readConfig(filename): 
    names = []
    types = []
    ranges = []
    with open(filename, "r") as fin: 
        lines = fin.readlines()
        for line in lines:
            line = line.strip()
            splited = line.split()
            if len(splited) < 3: 
                continue
            name = splited[0]
            typename = splited[1]
            values = splited[2:]
            for idx in range(len(values)): 
                if typename == "int": 
                    values[idx] = int(values[idx])
                elif typename == "float": 
                    values[idx] = float(values[idx])
            names.append(name)
            types.append(typename)
            ranges.append(values)
    return names, types, ranges

def runCommand(executable, configs, timeout=None, outfile=None): 
    command = [executable, ]
    for key, value in configs.items(): 
        command.append("--" + key)
        command.append(str(value))
    channel = sp.DEVNULL
    if not outfile is None: 
        channel = sp.PIPE
    result = 4
    try: 
        ret = sp.run(command, timeout=timeout, shell=False, stdout=channel, stderr=channel)
        if not outfile is None: 
            out = ret.stdout.decode("UTF-8")
            err = ret.stderr.decode("UTF-8")
            with open(outfile, "w") as fout: 
                if len(out) > 0: 
                    fout.write(out)
                    fout.write("\n")
                if len(err) > 0: 
                    fout.write(err)
                    fout.write("\n")
        result = ret.returncode
    except Exception: 
        pass
    return result

def runPythonCommand(filename, configs, timeout=None, outfile=None): 
    command = ["python3", filename, ]
    for key, value in configs.items(): 
        command.append("--" + key)
        command.append(str(value))
    channel = sp.DEVNULL
    if not outfile is None: 
        channel = sp.PIPE
    result = 4
    try: 
        ret = sp.run(command, timeout=timeout, shell=False, stdout=channel, stderr=channel)
        if not outfile is None: 
            out = ret.stdout.decode("UTF-8")
            err = ret.stderr.decode("UTF-8")
            with open(outfile, "w") as fout: 
                if len(out) > 0: 
                    fout.write(out)
                    fout.write("\n")
                if len(err) > 0: 
                    fout.write(err)
                    fout.write("\n")
        result = ret.returncode
    except Exception: 
        pass
    return result

def dominate(a, b): 
    assert len(a) == len(b)
    domin1 = True
    domin2 = False
    for idx in range(len(a)): 
        if a[idx] > b[idx]: 
            domin1 = False
        elif a[idx] < b[idx]: 
            domin2 = True
    return domin1 and domin2

def newParetoSet(paretoParams, paretoValues, newParams, newValue): 
    assert len(paretoParams) == len(paretoValues)
    dupli = False
    removed = set()
    indices = []
    for idx, elem in enumerate(paretoValues): 
        if str(paretoParams[idx]) == str(newParams): 
            dupli = True
            break
        if dominate(newValue, elem): 
            removed.add(idx)
    if dupli: 
        return paretoParams, paretoValues
    for idx, elem in enumerate(paretoValues): 
        if not idx in removed: 
            indices.append(idx)
    newParetoParams = []
    newParetoValues = []
    for index in indices: 
        newParetoParams.append(paretoParams[index])
        newParetoValues.append(paretoValues[index])
    bedominated = False
    for idx, elem in enumerate(newParetoValues): 
        if dominate(elem, newValue): 
            bedominated = True
    if len(removed) > 0:
        assert not bedominated
    if len(removed) > 0 or len(paretoParams) == 0 or not bedominated: 
        newParetoParams.append(newParams)
        newParetoValues.append(newValue)
    return newParetoParams, newParetoValues

def pareto(params, values): 
    paretoParams = []
    paretoValues = []

    for var, objs in zip(params, values): 
        paretoParams, paretoValues = newParetoSet(paretoParams, paretoValues, var, objs)

    return paretoParams, paretoValues


def _get_non_dominated(Y, maximize=True):
    is_efficient = torch.ones(
        *Y.shape[:-1],
        dtype=bool,
        device=Y.device
    )
    for i in range(Y.shape[-2]):
        i_is_efficient = is_efficient[..., i]
        if i_is_efficient.any():
            vals = Y[..., i : i + 1, :]
            if maximize:
                update = (Y > vals).any(dim=-1)
            else:
                update = (Y < vals).any(dim=-1)
            update[..., i] = i_is_efficient.clone()
            is_efficient2 = is_efficient.clone()
            if Y.ndim > 2:
                is_efficient2[~i_is_efficient] = False
            is_efficient[is_efficient2] = update[is_efficient2]
    return is_efficient


def get_non_dominated(Y, deduplicate=True):
    Y = torch.Tensor(Y)
    MAX_BYTES = 5e6
    n = Y.shape[-2]
    if n == 0:
        return torch.zeros(
            Y.shape[:-1],
            dtype=torch.bool,
            device=Y.device
        )
    el_size = 64 if Y.dtype == torch.double else 32
    if n > 1000 or \
        n ** 2 * Y.shape[:-2].numel() * el_size / 8 > MAX_BYTES:
        return _get_non_dominated(Y)

    Y1 = Y.unsqueeze(-3)
    Y2 = Y.unsqueeze(-2)
    dominates = (Y1 >= Y2).all(dim=-1) & (Y1 > Y2).any(dim=-1)
    nd_mask = ~(dominates.any(dim=-1))
    if deduplicate:
        indices = (Y1 == Y2).all(dim=-1).long().argmax(dim=-1)
        keep = torch.zeros_like(nd_mask)
        keep.scatter_(dim=-1, index=indices, value=1.0)
        return nd_mask & keep
    return nd_mask


if __name__ == "__main__": 
    paretoParams = [0, 1, 2, 3, 4, 5, 6, 7]
    paretoValues = [[70.76775036084953, 176.14678899082568, 72.04301075268818], [75.61802643938873, 125.6880733944954, 81.72043010752688], [70.04834238321075, 379.8165137614679, 66.66666666666667], [78.62625151785919, 141.28440366972475, 62.365591397849464], [81.19000160377574, 306.42201834862385, 50.53763440860215], [95.55754118267005, 106.42201834862385, 77.41935483870968], [82.08582491351066, 107.33944954128441, 60.215053763440864], [78.01681673425435, 117.43119266055047, 63.44086021505376]]

    newParam = 8
    newValue = [89.75416408916993, 97.24770642201834, 26.88172043010752]

    paretoParams, paretoValues = newParetoSet(paretoParams, paretoValues, newParam, newValue)
    print(paretoParams, paretoValues)
    




