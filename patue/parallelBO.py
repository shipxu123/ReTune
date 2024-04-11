import os
import sys 
sys.path.append(".")
import time
import multiprocessing as mp
from subprocess import run

import torch
import botorch
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.models.gp_regression import FixedNoiseGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement

from utils.utils import *

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DTYPE = torch.float if torch.cuda.is_available() else torch.double
DEVICE = torch.device("cpu")
DTYPE = torch.double

class ParallelBO: 

    def __init__(self, nDims, nObjs, nameVars, typeVars, rangeVars, funcEval, refpoint, scaleNoise, nInit=2**4, batchSize=2, numRestarts=10, rawSamples=512, mcSamples=256, nJobs=4): 
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
        self._scaleNoise = scaleNoise
        self._nInit = nInit
        self._batchSize = batchSize
        self._numRestarts = numRestarts
        self._rawSamples = rawSamples
        self._mcSamples = mcSamples
        self._nJobs = nJobs
        self._bounds = torch.tensor([[0.0] * self._nDims, [1.0] * self._nDims], \
                                    device=DEVICE, dtype=DTYPE)
        self._visited = {}

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
        score = self._funcEval(dict(zip(self._nameVars, values)))

        return score

    def evalPoint(self, point): 
        score = self._evalPoint(point).copy()
        for idx in range(len(score)): 
            score[idx] = -score[idx]

        return score

    def evalBatch(self, batch): 
        if isinstance(batch, list): 
            batch = torch.tensor(batch, device=DEVICE)
        results = []
        batch = batch.cpu().numpy()

        processes = []
        pool = mp.Pool(processes=self._nJobs)
        for jdx in range(batch.shape[0]): 
            param = batch[jdx]
            process = pool.apply_async(self.evalPoint, (param, ))
            processes.append(process)
            time.sleep(0.01)
        pool.close()
        pool.join()

        for idx in range(batch.shape[0]): 
            cost = processes[idx].get()
            results.append(cost)
        results = np.array(results)

        for idx in range(batch.shape[0]): 
            name = str(list(batch[idx]))
            if not name in self._visited: 
                self._visited[name] = list(-results[idx])
        
        return torch.tensor(results, device=DEVICE, dtype=DTYPE)
    

    def initSamples(self): 
        initX = draw_sobol_samples(bounds=self._bounds, n=self._nInit, q=1).squeeze(1)
        initY = self.evalBatch(initX)
        return initX, initY
    

    def initModel(self, trainX, trainY): 
        trainX = normalize(trainX, self._bounds)
        models = []
        for idx in range(self._nObjs):
            tmpY = trainY[..., idx:idx+1]
            tmpYvar = torch.full_like(tmpY, self._scaleNoise[idx] ** 2)
            models.append(
                FixedNoiseGP(trainX, tmpY, tmpYvar, outcome_transform=Standardize(m=1))
            )
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        return mll, model
    

    def getObservations(self, model, trainX):
        sampler = SobolQMCNormalSampler(num_samples=self._mcSamples)
        acqFunc = qNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=self._refpoint,  # use known reference point 
            X_baseline=normalize(trainX, self._bounds),
            prune_baseline=True,  # prune baseline points that have estimated zero probability of being Pareto optimal
            sampler=sampler,
        )
        candidates, _ = optimize_acqf(
            acq_function=acqFunc,
            bounds=torch.tensor([[0.0] * self._nDims, [1.0] * self._nDims], device=DEVICE, dtype=DTYPE),
            q=self._batchSize,
            num_restarts=self._numRestarts,
            raw_samples=self._rawSamples, 
            options={"batch_limit": 5, "maxiter": 200},
            sequential=True,
        )
        newX = unnormalize(candidates.detach(), bounds=self._bounds)
        newY = self.evalBatch(newX)
        newY = newY + torch.randn_like(newY) * torch.tensor(self._scaleNoise, device=DEVICE, dtype=DTYPE)
        
        return newX, newY

        
    def optimize(self, steps=2**4, verbose=True): 
        trainX, trainY = self.initSamples()
        mll, model = self.initModel(trainX, trainY)
        initX = trainX.cpu().numpy()
        initY = trainY.cpu().numpy()
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
            newX, newY = self.getObservations(model, trainX)
                    
            trainX = torch.cat([trainX, newX])
            trainY = torch.cat([trainY, newY])

            tmpBatchX = newX.cpu().numpy()
            tmpBatchY = newY.cpu().numpy()
            for param in list(tmpBatchX): 
                paretoParams, paretoValues = newParetoSet(paretoParams, paretoValues, list(param), self._evalPoint(param))

            mll, model = self.initModel(trainX, trainY)
            
            t1 = time.time()
            if verbose:
                print(f"Batch {iter:>2}: hypervolume:", calcHypervolume(self._refpoint, paretoValues)) 
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
    parser.add_argument("-j", "--njobs", required=False, action="store", default=4)
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
    
    def funcEval(config): 
        iter = str(time.time()).replace(".", "")
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
        
        return results
    
    names, types, ranges = readConfig(configfile)
    ndims = len(names)
    model = ParallelBO(ndims, nobjs, names, types, ranges, funcEval, refpoint=[refpoint] * nobjs, scaleNoise=[1.0] * nobjs, nInit=ninit, batchSize=int(args.batchsize), nJobs=int(args.njobs))
    results = model.optimize(steps=steps, verbose=True)
    print(results)
    list(map(lambda x: print("Parameter:", x[0], "\n -> Value:", x[1]), results))
    print("[Hypervolume]:", calcHypervolume([refpoint] * nobjs, list(map(lambda x: x[1], results))))
