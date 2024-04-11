import os
import sys 
sys.path.append(".")
import time
import random
import multiprocessing as mp

from os.path import abspath
from datetime import datetime
from multiprocessing import cpu_count
from subprocess import run

import sobol_seq
from botorch.utils.sampling import draw_sobol_samples
import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as nnfunc
from torch.utils.data import DataLoader


from utils.utils import *
from utils.littleNN import *

DEVICE = torch.device("cpu")
DTYPE = torch.double

POOLSIZE = 32

class Recommend: 

    def __init__(self, nDims, nObjs, nameVars, typeVars, rangeVars, funcEval, weights, refpoint, sizeInter=16, nInit=2**4): 
        self._nDims      = nDims
        self._nObjs      = nObjs
        self._nameVars   = nameVars
        self._typeVars   = typeVars
        self._rangeVars  = rangeVars
        self._funcEval   = funcEval
        self._name2index = {}
        for idx, name in enumerate(self._nameVars): 
            self._name2index[name] = idx
        
        self._weights = weights
        self._refpoint = [refpoint] * self._nObjs
        self._nInit = nInit
        self._sizeInter = sizeInter
        self._bound = [0.0, 1.0]
        self._bounds = [self._bound, ] * self._nDims
        # self._grid = sobol_seq.i4_sobol_generate(self._nDims, nInit*4, np.random.randint(0, nInit))
        bounds = torch.tensor([[0.0] * self._nDims, [1.0] * self._nDims], device=DEVICE, dtype=DTYPE)
        self._grid = draw_sobol_samples(bounds=bounds, n=1024*4, q=1).squeeze(1).cpu().numpy()
        self._visited = {}

    def objective(self, variables): 
        values = []
        for idx, name in enumerate(self._nameVars): 
            typename = self._typeVars[idx]
            value = None
            if typename == "int": 
                value = round(variables[idx] * (self._rangeVars[idx][1] - self._rangeVars[idx][0])) + self._rangeVars[idx][0]
            elif typename == "float": 
                value = variables[idx] * (self._rangeVars[idx][1] - self._rangeVars[idx][0]) + self._rangeVars[idx][0]
            elif typename == "enum": 
                value = self._rangeVars[idx][round(variables[idx] * (len(self._rangeVars[idx]) - 1))] if len(self._rangeVars[idx]) > 1 else self._rangeVars[idx][0]
            else: 
                assert typename in ["int", "float", "enum"]
            assert not value is None
            values.append(value)
        name = str(variables)
        if name in self._visited: 
            return self._visited[name]
        config = dict(zip(self._nameVars, values))
        score = self._funcEval(config)
        self._visited[name] = score
        print("[Eval] Point:", config)
        print(" -> Result:", score)
        with open("historyRecommend.txt", "a+") as fout: 
            fout.write(str(config) + "\n")
            fout.write(str(score) + "\n")
        return score

    def _evalPoint(self, point): 
        values = []
        for idx, name in enumerate(self._nameVars): 
            typename = self._typeVars[idx]
            value = None
            if typename == "int": 
                value = int(round(point[idx] * (self._rangeVars[idx][1] - self._rangeVars[idx][0])) + self._rangeVars[idx][0])
            elif typename == "float": 
                value = point[idx] * (self._rangeVars[idx][1] - self._rangeVars[idx][0]) + self._rangeVars[idx][0]
            elif typename == "enum": 
                value = self._rangeVars[idx][int(round(point[idx] * (len(self._rangeVars[idx]) - 1)))] if len(self._rangeVars[idx]) > 1 else self._rangeVars[idx][0]
            else: 
                assert typename in ["int", "float", "enum"]
            assert not value is None
            values.append(value)
        name = str(list(point))
        if name in self._visited: 
            return self._visited[name]
        config = dict(zip(self._nameVars, values))
        score = self._funcEval(config)
        print("[Eval] Point:", config)
        print(" -> Result:", score)
        with open("historyRecommend.txt", "a+") as fout: 
            fout.write(str(config) + "\n")
            fout.write(str(score) + "\n")

        return score

    def evalPoint(self, point): 
        score = self._evalPoint(point).copy()
        for idx in range(len(score)): 
            score[idx] = score[idx]

        return score

    def evalBatch(self, batch): 
        if not isinstance(batch, torch.Tensor): 
            batch = torch.tensor(batch, dtype=DTYPE, device=DEVICE)
        results = []
        batch = batch.cpu().numpy()
        while batch.shape[0] % POOLSIZE != 0: 
            batch = np.concatenate([batch, batch[-1:]], axis=0)

        for idx in range(0, batch.shape[0], POOLSIZE): 
            processes = []
            pool = mp.Pool(processes=POOLSIZE)
            for jdx in range(POOLSIZE): 
                param = batch[idx + jdx]
                process = pool.apply_async(self.evalPoint, (param, ))
                processes.append(process)
                time.sleep(0.01)
            pool.close()
            pool.join()

            for jdx in range(POOLSIZE): 
                cost = processes[jdx].get()
                results.append(cost)
        results = np.array(results)
        
        return results
    
    def init(self): 
        params = []
        objs = []
        index = np.random.randint(0, self._grid.shape[0])
        for idx in range(self._nInit):
            print("[Init] Getting sample", idx)
            exist = True
            while exist:
                index = np.random.randint(0, self._grid.shape[0])
                x_rand = self._grid[index : (index + 1), :][0]
                if (any((x_rand == x).all() for x in params)) == False:
                    exist = False
            point = list(x_rand)
            params.append(point)
        objs = self.evalBatch(params)

        return np.array(params, dtype=np.float32), np.array(objs, dtype=np.float32)
        
    def optimize(self, divide=1.0, samples=2**4, nMC=1024, batchSize=2**4, epochs=4): 
        trainX, trainY = self.init()
        trainset = list(zip(trainX, trainY / divide))
        loaderTrain = torch.utils.data.DataLoader(trainset, batch_size=batchSize)
        
        params = []
        values = []
        for idx, param in enumerate(list(trainX)): 
            params.append(list(param))
            values.append(list(trainY[idx]))
        paretoParams, paretoValues = pareto(params, values)
        print(f'[Initial PARETO]: {paretoParams}, {paretoValues}')
        print(f'[Hypervolume]:', calcHypervolume(self._refpoint, paretoValues))

        model = LittleNN(sizeInput=self._nDims, sizeInter=self._sizeInter, sizeOutput=self._nObjs).to(DEVICE)
        print(model)
        model.trainNet(loaderTrain, epochs=epochs)

        testX = np.random.rand(nMC, self._nDims).astype(np.float32)
        pred  = model.predX(testX)
        score = np.sum(pred * np.array(self._weights)[np.newaxis, :], axis=1)
        indices = np.argsort(score)[:samples]
        testX = testX[indices]
        scores = self.evalBatch(testX)
        points = []
        results = []
        for index in range(len(testX)): 
            point = list(testX[index])
            score = scores[index]
            points.append(point)
            results.append(score)
        paretoParams, paretoValues = pareto(points, results)

        # index = np.argmin(results)
        # return [[points[index], results[index]], ]
        return list(zip(paretoParams, paretoValues))

import argparse
def parseArgs(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--paramfile", required=True, action="store")
    parser.add_argument("-c", "--command", required=True, action="store")
    parser.add_argument("-r", "--refpoint", required=True, action="store")
    parser.add_argument("-d", "--divide", required=False, action="store", default=1000.0)
    parser.add_argument("-n", "--nobjs", required=True, action="store")
    parser.add_argument("-b", "--batchsize", required=False, action="store", default=16)
    parser.add_argument("-e", "--epochs", required=False, action="store", default=16)
    parser.add_argument("-i", "--ninit", required=False, action="store", default=16)
    parser.add_argument("-s", "--steps", required=False, action="store", default=64)
    parser.add_argument("-t", "--timeout", required=False, action="store", default=None)
    parser.add_argument("-o", "--output", required=False, action="store", default="tmp")
    return parser.parse_args()


if __name__ == "__main__": 
    print("GO!")
    args = parseArgs()
    configfile = args.paramfile
    command = args.command
    refpoint = float(args.refpoint)
    nobjs = int(args.nobjs)
    ninit = int(args.ninit)
    steps = int(args.steps)
    timeout = None if args.timeout is None else float(args.timeout)
    folder = args.output
    if not os.path.exists(folder): 
        os.mkdir(folder)
    run = runCommand
    if command[-3:] == ".py": 
        run = runPythonCommand
    
    iter = 0
    def funcEval(config): 
        global iter
        filename = folder + f"/run{iter}.log"
        ret = run(command, config, timeout, filename)
        results = [refpoint, ] * nobjs
        try: 
            with open(filename, "r") as fin: 
                lines = fin.readlines()
                splited = lines[0].split()
                for idx, elem in enumerate(splited): 
                    if len(elem) > 0 and idx < len(results): 
                        results[idx] = float(elem)
        except Exception: 
            pass
        iter += 1
        
        return results
    
    names, types, ranges = readConfig(configfile)
    print("[Params]", names)
    ndims = len(names)
    model = Recommend(ndims, nobjs, names, types, ranges, funcEval, refpoint=refpoint, weights=[1.0/nobjs] * nobjs, sizeInter=16, nInit=ninit)
    results = model.optimize(divide=float(args.divide), samples=steps, nMC=steps*4, batchSize=int(args.batchsize), epochs=int(args.epochs))
    list(map(lambda x: print("Parameter:", x[0], "\n -> Value:", x[1], "\n -> ", model.objective(x[0])), results))
    print("[Hypervolume]:", calcHypervolume([refpoint] * nobjs, list(map(lambda x: x[1], results))))
    
