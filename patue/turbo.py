import os
import sys 
sys.path.append(".")
import math
import time
import multiprocessing as mp
from dataclasses import dataclass

import torch
from botorch.acquisition import qExpectedImprovement
from botorch.fit import fit_gpytorch_model
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.test_functions import Ackley
from botorch.utils.transforms import unnormalize
from torch.quasirandom import SobolEngine

import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import HorseshoePrior

from utils.utils import *

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DTYPE = torch.float if torch.cuda.is_available() else torch.double
DEVICE = torch.device("cpu")
DTYPE = torch.double

def samplesSobol(ndims, samples, seed=None):
    sobol = SobolEngine(dimension=ndims, scramble=True, seed=seed)
    return sobol.draw(n=samples).to(dtype=DTYPE, device=DEVICE)

@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 10  # Note: The original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )

    def update(self, testY):
        if max(testY) > self.best_value + 1e-3 * math.fabs(self.best_value):
            self.success_counter += 1
            self.failure_counter = 0
        else:
            self.success_counter = 0
            self.failure_counter += 1

        if self.success_counter == self.success_tolerance:  # Expand trust region
            self.length = min(2.0 * self.length, self.length_max)
            self.success_counter = 0
        elif self.failure_counter == self.failure_tolerance:  # Shrink trust region
            self.length /= 2.0
            self.failure_counter = 0

        self.best_value = max(self.best_value, max(testY).item())
        if self.length < self.length_min:
            self.restart_triggered = True


class Turbo: 

    def __init__(self, nDims, nObjs, nameVars, typeVars, rangeVars, funcEval, refpoint, weights, nInit=2**4, batchSize=2, mcSamples=256, nJobs=4): 
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
        self._weights = weights
        self._nInit = nInit
        self._batchSize = batchSize
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
        
        return torch.sum(torch.tensor(results) * torch.tensor(self._weights, device=DEVICE, dtype=DTYPE).unsqueeze(0), dim=1, keepdim=True)
    

    def initSamples(self): 
        initX = samplesSobol(self._nDims, self._nInit)
        initY = self.evalBatch(initX)
        return initX, initY
    

    def initModel(self, trainX, normY): 
        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
            MaternKernel(nu=2.5, ard_num_dims=self._nDims, lengthscale_constraint=Interval(0.005, 4.0))
        )
        model = SingleTaskGP(trainX, normY, covar_module=covar_module, likelihood=likelihood)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        
        return mll, model


    def getObservations(self, state, model, trainX, trainY, n_candidates=None):
        if n_candidates is None:
            n_candidates = min(5000, max(2000, 200 * trainX.shape[-1]))

        # Scale the TR to be proportional to the lengthscales
        center = trainX[trainY.argmax(), :].clone()
        weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
        weights = weights / weights.mean()
        weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
        tr_lb = torch.clamp(center - weights * state.length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(center + weights * state.length / 2.0, 0.0, 1.0)

        dim = trainX.shape[-1]
        sobol = SobolEngine(dim, scramble=True)
        pert = sobol.draw(n_candidates).to(dtype=DTYPE, device=DEVICE)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = (
            torch.rand(n_candidates, dim, dtype=DTYPE, device=DEVICE)
            <= prob_perturb
        )
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=DEVICE)] = 1

        # Create candidate points from the perturbations and the mask        
        X_cand = center.expand(n_candidates, dim).clone()
        X_cand[mask] = pert[mask]

        # Sample on the candidate points
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
        with torch.no_grad():  # We don't need gradients when using TS
            testX = thompson_sampling(X_cand, num_samples=self._batchSize)

        newX = testX.detach()
        newY = self.evalBatch(newX)

        return newX, newY

        
    def optimize(self, steps=2**4, verbose=True): 
        trainX, trainY = self.initSamples()
        normY = (trainY - trainY.mean()) / trainY.std()
        state = TurboState(dim=self._nDims, batch_size=self._batchSize)

        initX = trainX.cpu().numpy()
        params = []
        values = []
        for param in list(initX): 
            params.append(list(param))
            values.append(self._evalPoint(param))
        paretoParams, paretoValues = pareto(params, values)
        print(f'[Initial PARETO]: {paretoParams}, {paretoValues}')
        
        for iter in range(steps):  
            mll, model = self.initModel(trainX, normY)
            with gpytorch.settings.max_cholesky_size(float("inf")):
                fit_gpytorch_model(mll)
                newX, newY = self.getObservations(state, model, trainX, normY)
            trainX = torch.cat([trainX, newX])
            trainY = torch.cat([trainY, newY])
            normY = (trainY - trainY.mean()) / trainY.std()

            tmpBatchX = newX.cpu().numpy()
            for param in list(tmpBatchX): 
                paretoParams, paretoValues = newParetoSet(paretoParams, paretoValues, list(param), self._evalPoint(param))

            state.update(newY)
                
            t1 = time.time()
            if verbose:
                print(self._refpoint, paretoValues)
                print(f"Batch {iter:>2}: hypervolume:", calcHypervolume(self._refpoint, paretoValues)) 
            else:
                print(".", end="")

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
    model = Turbo(ndims, nobjs, names, types, ranges, funcEval, refpoint=refpoint, weights=[1.0/nobjs] * nobjs, nInit=ninit, batchSize=int(args.batchsize))
    results = model.optimize(steps=steps, verbose=True)
    print(results)
    list(map(lambda x: print("Parameter:", x[0], "\n -> Value:", x[1]), results))
    print("[Hypervolume]:", calcHypervolume([refpoint] * nobjs, list(map(lambda x: x[1], results))))

