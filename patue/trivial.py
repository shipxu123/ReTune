import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ARDRegression, LinearRegression
import torch
from utils.utils import *
from utils.littleNN import *
DEVICE = torch.device("cpu")
DTYPE = torch.float

if __name__ == "__main__":
    candidates = []
    names = []
    infos = []
    results = []
    with open("historyREMORBO2index1.txt", "r") as fin: 
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
            if isinstance(key, str): 
                number = values[jdx][key]
                infos[idx][jdx] = number
    infos = np.array(infos)
    results = np.array(results)

    REFS = [150.0, 150.0, 150.0]

    trainX, trainY = infos.astype(np.float32), results.astype(np.float32)
    trainXmin = np.min(trainX, axis=0, keepdims=True)
    trainXmax = np.max(trainX, axis=0, keepdims=True)
    trainX = (trainX - trainXmin) / (trainXmax - trainXmin)
    trainset = list(zip(trainX, trainY / 1000.0))
    loaderTrain = torch.utils.data.DataLoader(trainset, batch_size=16)
    
    params = []
    values = []
    for idx, param in enumerate(list(trainX)): 
        params.append(list(param))
        values.append(list(trainY[idx]))
    paretoParams, paretoValues = pareto(params, values)
    print(paretoValues)
    print(f'[Hypervolume]:', calcHypervolume(REFS, paretoValues))

    NDIMS = len(paretoParams[0])
    model = LittleNN(sizeInput=NDIMS, sizeInter=16, sizeOutput=3).to(DEVICE)
    print(model)
    model.trainNet(loaderTrain, epochs=256)

    NMC = 4096
    WEIGHTS = np.array([1/3.0, 1/3.0, 1/3.0])
    testX = np.random.rand(NMC, NDIMS).astype(np.float32)
    pred  = model.predX(testX)
    score = np.sum(pred * WEIGHTS[np.newaxis, :], axis=1)
    indices = np.argsort(score)[:256]
    points = []
    results = []
    for index in indices: 
        point = list(testX[index])
        score = pred[index] * 1000.0
        result = np.sum(np.array(score) * np.array(WEIGHTS))
        points.append(point)
        results.append(result)
        paretoParams, paretoValues = newParetoSet(paretoParams, paretoValues, point, score)

    print(paretoValues)
    print("[Hypervolume]:", calcHypervolume(REFS, paretoValues))