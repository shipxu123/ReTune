import os
import sys 
sys.path.append(".")
import time
import random
from itertools import combinations

from os.path import abspath
from datetime import datetime
from multiprocessing import cpu_count
from subprocess import run

import torch
import botorch
from botorch.models import SingleTaskGP
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood, ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
from botorch.optim import optimize_acqf
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.exceptions import BadInitialCandidatesWarning

from botorch.models.transforms.input import Normalize
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption

from utils.utils import *

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DTYPE = torch.float if torch.cuda.is_available() else torch.double
DEVICE = torch.device("cpu")
DTYPE = torch.double

class VanillaBO: 

    def __init__(self, nDims, nObjs, nameVars, typeVars, rangeVars, funcEval, weights, refpoint=0.0, nInit=2**4, batchSize=2, numRestarts=10, rawSamples=512, mcSamples=256): 
        self._nDims      = nDims
        self._nObjs      = nObjs
        self._nameVars   = nameVars
        self._typeVars   = typeVars
        self._rangeVars  = rangeVars
        self._funcEval   = funcEval
        self._name2index = {}
        for idx, name in enumerate(self._nameVars): 
            self._name2index[name] = idx
        self._refpoint = [refpoint] * self._nObjs
        self._nInit = nInit
        self._batchSize = batchSize
        self._weights = weights
        self._numRestarts = numRestarts
        self._rawSamples = rawSamples
        self._mcSamples = mcSamples
        self._bounds = torch.tensor([[0.0] * self._nDims, [1.0] * self._nDims], \
                                    device=DEVICE, dtype=DTYPE)
        self._visited = {}
        self.n_comp = 1

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
        name = str(point)
        if name in self._visited: 
            return self._visited[name]
        config = dict(zip(self._nameVars, values))
        score = self._funcEval(config)
        self._visited[name] = score
        print("[Eval] Point:", config)
        print(" -> Result:", score)
        with open("historyBO.txt", "a+") as fout: 
            fout.write(str(config) + "\n")
            fout.write(str(score) + "\n")

        return score

    def evalPoint(self, point): 
        score = self._evalPoint(point)
        return score
        #assert len(score) <= len(self._weights)
        #cost = 0.0
        #for idx, elem in enumerate(score): 
        #    cost += self._weights[idx] * elem

        #return -cost

    def evalBatch(self, batch): 
        if isinstance(batch, list): 
            batch = torch.tensor(batch, device=DEVICE)
        results = []
        batch = batch.cpu().numpy()
        for idx in range(batch.shape[0]): 
            param = batch[idx]
            cost = self.evalPoint(param)
            results.append(cost)
        
        results = torch.tensor(results, device=DEVICE)

        # minimize power and area
        results[:, 1] *= -1.0
        results[:, 2] *= -1.0

        return results

    def initSamples(self): 
        initX = torch.rand(self._nInit, self._nDims, device=DEVICE, dtype=DTYPE)
        initY = self.evalBatch(initX)
        return initX, initY

    def _generate_comparisons(self, y, n_comp, noise=0.0, replace=False):
        """Create pairwise comparisons with noise(non-dominate)"""
        # generate all possible pairs of elements in y
        all_pairs = np.array(list(combinations(range(y.shape[0]), 2)))
        # randomly select n_comp pairs from all_pairs
        comp_pairs = all_pairs[
            np.random.choice(range(len(all_pairs)), n_comp, replace=replace)
        ]
        # add gaussian noise to the latent y values
        c0 = y[comp_pairs[:, 0]] + np.random.standard_normal(len(comp_pairs)) * noise
        c1 = y[comp_pairs[:, 1]] + np.random.standard_normal(len(comp_pairs)) * noise
        # reverse_comp = (c0 < c1).numpy()
        reverse_comp = (c0 < c1).numpy().all(axis=-1)
        comp_pairs[reverse_comp, :] = np.flip(comp_pairs[reverse_comp, :], 1)
        comp_pairs = torch.tensor(comp_pairs).long()
    
        return comp_pairs

    def initModel(self, trainX, trainY): 
        models = []
        comparisons = self._generate_comparisons(trainY, n_comp=self.n_comp, noise=0.0)
        model = PairwiseGP(
            trainX,
            comparisons,
            input_transform=Normalize(d=trainX.shape[-1]),
        )
        mll = PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model)
        return mll, model


    def getObservations(self, acqFunc):
        # optimize
        # q = 2 number of points per query
        candidates, _ = optimize_acqf(
            acq_function=acqFunc,
            bounds=self._bounds,
            q=2,
            num_restarts=self._numRestarts,
            raw_samples=self._rawSamples,  # used for intialization heuristic
        )

        # observe new values 
        newX = candidates.detach()
        newY = self.evalBatch(newX)
        return newX, newY

    def optimize(self, steps=2**4, verbose=True): 
        bestX = None
        bestY = None

        trainX, trainY = self.initSamples()
        mll, model = self.initModel(trainX, trainY)
        initX = trainX.cpu().numpy()
        initY = trainY.cpu().numpy()
        #index = initY.argmax()
        index = get_non_dominated(initY).numpy()
        print(f"index: {index}")
        print(f"initY Shape : {initY.shape}")
        bestX = initX[index]
        bestY = initY[index]

        params = []
        values = []
        for param in list(initX): 
            params.append(list(param))
            values.append(self._evalPoint(param))
        paretoParams, paretoValues = pareto(params, values)
        print(f'[Initial PARETO]: {paretoParams}, {paretoValues}')
        print(f'[Hypervolume]:', calcHypervolume(self._refpoint, paretoValues))

        for iter in range(steps):
            t0 = time.time()

            fit_gpytorch_model(mll)

            acq_func = AnalyticExpectedUtilityOfBestOption(pref_model=model)
            newX, newY = self.getObservations(acq_func)

            trainX = torch.cat([trainX, newX])
            trainY = torch.cat([trainY, newY])

            tmpX = trainX.cpu().numpy()
            tmpY = trainY.cpu().numpy()
            #index = tmpY.argmax()
            index = get_non_dominated(tmpY).numpy()
            bestX = tmpX[index]
            bestY = tmpY[index]

            tmpBatchX = newX.cpu().numpy()
            tmpBatchY = newY.cpu().numpy()
            #index = tmpBatchY.argmax()
            index = get_non_dominated(tmpBatchY).numpy()
            bestBatchX = tmpBatchX[index]
            bestBatchY = tmpBatchY[index]

            for param in list(tmpBatchX): 
                paretoParams, paretoValues = newParetoSet(paretoParams, paretoValues, list(param), self._evalPoint(param))

            mll, model = self.initModel(trainX, trainY)
            
            t1 = time.time()
            if verbose:
                print(f"Batch {iter:>2}: best params = {bestX}; best result = {-bestY}")
                print(f" -> Batch best = {bestBatchX}, {-bestBatchY}")
                print(f" -> Pareto-front:", paretoValues)
                print(f'[Hypervolume]:', calcHypervolume(self._refpoint, paretoValues))
            else:
                print(".", end="")

        # return [[bestX, -bestY], ]
        return list(zip(paretoParams, paretoValues))


import argparse
def parseArgs(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--paramfile", required=True, action="store")
    parser.add_argument("-c", "--command", required=True, action="store")
    parser.add_argument("-r", "--refpoint", required=True, action="store")
    parser.add_argument("-n", "--nobjs", required=True, action="store")
    parser.add_argument("-b", "--batchsize", required=False, action="store", default=4)
    parser.add_argument("-i", "--ninit", required=False, action="store", default=16)
    parser.add_argument("-s", "--steps", required=False, action="store", default=64)
    parser.add_argument("-t", "--timeout", required=False, action="store", default=None)
    parser.add_argument("-o", "--output", required=False, action="store", default="tmp")
    return parser.parse_args()


if __name__ == "__main__": 
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
    #def funcEval(config): 
    #    global iter
    #    filename = folder + f"/run{iter}.log"
    #    ret = run(command, config, timeout, filename)
    #    results = [refpoint, ] * nobjs
    #    try: 
    #        with open(filename, "r") as fin: 
    #            lines = fin.readlines()
    #            splited = lines[0].split()
    #            for idx, elem in enumerate(splited): 
    #                if len(elem) > 0 and idx < len(results): 
    #                    results[idx] = float(elem)  
    #    except Exception: 
    #        pass
    #    iter += 1
    #    
    #    return results
    def funcEval(config):
        global iter
        results = [random.random() * 150  for _ in range(nobjs)]
        iter += 1
        return results

    names, types, ranges = readConfig(configfile)
    ndims = len(names)
    model = VanillaBO(ndims, nobjs, names, types, ranges, funcEval, refpoint=refpoint, weights=[1.0/nobjs] * nobjs, nInit=ninit, batchSize=int(args.batchsize))
    results = model.optimize(steps=steps, verbose=True)
    print(results)
    list(map(lambda x: print("Parameter:", x[0], "\n -> Value:", x[1]), results))
    print("[Hypervolume]:", calcHypervolume([refpoint] * nobjs, list(map(lambda x: x[1], results))))
