import os
import sys 
sys.path.append(".")
import math
import time
import random
import multiprocessing as mp
from dataclasses import dataclass

from sklearn.cluster import KMeans

import torch
from botorch.acquisition import qExpectedImprovement
from botorch.fit import fit_gpytorch_model
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP, ModelList
from botorch.optim import optimize_acqf
from botorch.test_functions import Ackley
from botorch.utils.transforms import unnormalize
from torch.quasirandom import SobolEngine

import gpytorch
import gpytorch.settings as gpts
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel, RFFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import HorseshoePrior
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

from utils.utils import *

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DTYPE = torch.float if torch.cuda.is_available() else torch.double
DEVICE = torch.device("cpu")
DTYPE = torch.double

def samplesSobol(ndims, samples, seed=None):
    sobol = SobolEngine(dimension=ndims, scramble=True, seed=seed)
    return sobol.draw(n=samples).to(dtype=DTYPE, device=DEVICE)

@dataclass
class MorboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.01
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = 2
    success_counter: int = 0
    success_tolerance: int = 5
    restart_triggered: bool = False

    def update(self, success):
        if success:
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

        if self.length < self.length_min:
            self.restart_triggered = True


class Morbo: 

    def __init__(self, nDims, nObjs, nameVars, typeVars, rangeVars, funcEval, refpoint, weights, nInit=2**4, batchSize=2, mcSamples=1024, nJobs=4): 
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
        config = dict(zip(self._nameVars, values))
        score = self._funcEval(config)
        print("[Eval] Point:", config)
        print(" -> Result:", score)
        with open("historyMORBO.txt", "a+") as fout: 
            fout.write(str(config) + "\n")
            fout.write(str(score) + "\n")

        return score

    def evalPoint(self, point): 
        score = self._evalPoint(point).copy()
        for idx in range(len(score)): 
            score[idx] = -score[idx]

        return score

    def evalBatch(self, batch): 
        if not isinstance(batch, torch.Tensor): 
            batch = torch.tensor(batch, dtype=DTYPE, device=DEVICE)
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
    

    def initSamples(self, ninit=None): 
        if ninit is None: 
            ninit = self._nInit
        initX = samplesSobol(self._nDims, ninit)
        initY = self.evalBatch(initX)
        return initX, initY
    

    def initModel(self, trainX, trainY): 
        models = []
        for idx in range(self._nObjs):
            tmpY = trainY[..., idx:idx+1]
            likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
            covar_module = ScaleKernel(
                MaternKernel(nu=2.5, ard_num_dims=self._nDims)
            )
            model = SingleTaskGP(trainX, tmpY, covar_module=covar_module, likelihood=likelihood)
            models.append(model)
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        return mll, model


    def getRegionHV(self, index):
        print("[getRegionPareto]", index)
        state = self._states[index]
        trainX = self._dataXY[index][0]
        trainY = self._dataXY[index][1]

        initX = trainX.cpu().numpy()
        initY = trainY.cpu().numpy()
        initParams = []
        initValues = []
        for idx, param in enumerate(list(initX)): 
            initParams.append(list(param))
            initValues.append(list(-initY[idx]))
        initParetoParams, initParetoValues = pareto(initParams, initValues)
        initHV = calcHypervolume(self._refpoint, initParetoValues)

        return initHV


    def getObservations(self, index):
        print("[GetObservations]", index)
        state = self._states[index]
        trainX = self._dataXY[index][0]
        trainY = self._dataXY[index][1]

        # Select the center
        hvc = []

        initX = trainX.cpu().numpy()
        initY = trainY.cpu().numpy()
        initParams = []
        initValues = []
        for idx, param in enumerate(list(initX)): 
            initParams.append(list(param))
            initValues.append(list(-initY[idx]))
        initParetoParams, initParetoValues = pareto(initParams, initValues)
        print(initParetoValues)
        initHV = calcHypervolume(self._refpoint, initParetoValues)

        for idx in range(initX.shape[0]): 
            newParams = []
            newValues = []
            for jdx, param in enumerate(list(initX)): 
                if idx == jdx: 
                    continue
                newParams.append(list(param))
                newValues.append(list(-initY[jdx]))
            newParetoParams, newParetoValues = pareto(newParams, newValues)
            newHV = calcHypervolume(self._refpoint, newParetoValues)
            newHVC = initHV - newHV
            hvc.append(newHVC)
        hvc = np.array(hvc)

        center = trainX[hvc.argmax(), :].clone()
        tr_lb = torch.clamp(center - state.length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(center + state.length / 2.0, 0.0, 1.0)
        x_lb = torch.clamp(center - state.length, 0.0, 1.0)
        x_ub = torch.clamp(center + state.length, 0.0, 1.0)

        # Fit the model
        print("[Center]:", center)
        dataX = trainX
        dataY = trainY
        used = set()
        for idx in range(dataX.shape[0]): 
            used.add(str(dataX[idx]))
        indices = []
        for idx in range(self._dataAllX.shape[0]): 
            if torch.all(self._dataAllX[idx] <= x_ub) and torch.all(self._dataAllX[idx] >= x_lb) and not str(self._dataAllX[idx]) in used: 
                indices.append(idx)
                used.add(str(self._dataAllX[idx]))
        dataX = torch.cat([dataX, self._dataAllX[indices]])
        dataY = torch.cat([dataY, self._dataAllY[indices]])

        mll, model = self.initModel(dataX, dataY)
        fit_gpytorch_model(mll)
            

        # Sample on the candidate points
        predX = None
        for submodel in model.models: 
            dim = trainX.shape[-1]
            sobol = SobolEngine(dim, scramble=True)
            candX = sobol.draw(1024 * 4).to(dtype=DTYPE, device=DEVICE)
            candX = tr_lb + (tr_ub - tr_lb) * candX
            with gpts.fast_computations(covar_root_decomposition=True): 
                with torch.no_grad():  # We don't need gradients when using TS
                    thompson_sampling = MaxPosteriorSampling(model=submodel, replacement=False)
                    tmp = thompson_sampling(candX, num_samples=1024)
                if predX is None: 
                    predX = tmp
                else: 
                    predX = torch.cat([predX, tmp],)
        predY = torch.zeros([predX.shape[0], self._nObjs], dtype=DTYPE, device=DEVICE)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for idx, submodel in enumerate(model.models): 
                submodel.eval()
                mll.eval()
                predY[:, idx] = mll.likelihood(submodel(predX))[0].sample()
        predX = predX.cpu().numpy()
        predY = predY.cpu().numpy()
        hvi = []
        for idx in range(predX.shape[0]): 
            tmp1 = list(predX[idx])
            tmp2 = list(-predY[idx])
            newParetoParams, newParetoValues = newParetoSet(initParetoParams, initParetoValues, tmp1, tmp2)
            newHV = calcHypervolume(self._refpoint, newParetoValues)
            hvi.append(newHV - initHV)
        indices = list(range(len(hvi)))
        indices = sorted(indices, key=lambda x: (-hvi[x], -np.sum(np.abs(predY[x] - np.array(self._refpoint)))))
        hvi = np.array(hvi)
        indices = indices[:self._batchSize]
        print(" -> [Sampling]:", indices, "HVI:", hvi[indices])

        newX = torch.tensor(predX[indices], dtype=DTYPE, device=DEVICE)
        newY = self.evalBatch(newX)

        return newX, newY

        
    def optimize(self, steps=2**4, regions=4, verbose=True): 
        self._states = []
        self._mlls   = []
        self._models = []
        self._dataXY = []
        self._dataAllX = None
        self._dataAllY = None

        # Initialize all regions
        trainAllX, trainAllY = self.initSamples(regions * self._nInit)
        trainAllX = trainAllX.cpu().numpy()
        trainAllY = trainAllY.cpu().numpy()
        kmeans = KMeans(n_clusters=regions)
        indices = kmeans.fit_predict(trainAllX)
        trainXs = [[] for _ in range(regions)]
        trainYs = [[] for _ in range(regions)]
        for idx in range(regions): 
            for jdx, region in enumerate(indices): 
                if idx == region: 
                    trainXs[idx].append(trainAllX[jdx])
                    trainYs[idx].append(trainAllY[jdx])
            assert len(trainXs[idx]) > 0
            assert len(trainYs[idx]) > 0
            trainXs[idx] = np.array(trainXs[idx])
            trainYs[idx] = np.array(trainYs[idx])
            trainXs[idx] = torch.tensor(trainXs[idx], dtype=DTYPE, device=DEVICE)
            trainYs[idx] = torch.tensor(trainYs[idx], dtype=DTYPE, device=DEVICE)

        initParams = []
        initValues = []
        for region in range(regions): 
            trainX, trainY = trainXs[region], trainYs[region]
            state = MorboState(dim=self._nDims, batch_size=self._batchSize)
            initX = trainX.cpu().numpy()
            initY = trainY.cpu().numpy()
            for idx in range(initX.shape[0]): 
                initParams.append(list(initX[idx]))
                initValues.append(list(-initY[idx]))
            self._states.append(state)
            self._dataXY.append([trainX, trainY])
            if self._dataAllX is None: 
                self._dataAllX = trainX
            else: 
                self._dataAllX = torch.cat([self._dataAllX, trainX])
            if self._dataAllY is None: 
                self._dataAllY = trainY
            else: 
                self._dataAllY = torch.cat([self._dataAllY, trainY])
        paretoParams, paretoValues = pareto(initParams, initValues)
        print(f'[Initial PARETO]: {paretoParams}, {paretoValues}')
        print(f'[Hypervolume]:', calcHypervolume(self._refpoint, paretoValues))
        
        for iter in range(steps): 
            for region in range(regions): 
                with gpytorch.settings.max_cholesky_size(float("inf")):
                    newX, newY = self.getObservations(region)
                self._dataXY[region][0] = torch.cat([self._dataXY[region][0], newX])
                self._dataXY[region][1] = torch.cat([self._dataXY[region][1], newY])
                self._dataAllX = torch.cat([self._dataAllX, newX])
                self._dataAllY = torch.cat([self._dataAllY, newY])

                paretoPrev = str(paretoValues)
                tmpX = newX.cpu().numpy()
                tmpY = newY.cpu().numpy()
                for idx in range(tmpX.shape[0]): 
                    paretoParams, paretoValues = newParetoSet(paretoParams, paretoValues, list(tmpX[idx]), list(-tmpY[idx]))
                paretoPost = str(paretoValues)

                self._states[region].update(paretoPrev != paretoPost)
                regionHV = self.getRegionHV(region)
                print("[RegionHV]", regionHV)
                if self._states[region].restart_triggered or regionHV == 0.0: 
                    print(f"[Restart] region {region} restart")
                    trainX, trainY = self.initSamples()
                    self._states[region] = MorboState(dim=self._nDims, batch_size=self._batchSize)
                    self._dataXY[region][0] = trainX
                    self._dataXY[region][1] = trainY
                    self._dataAllX = torch.cat([self._dataAllX, trainX])
                    self._dataAllY = torch.cat([self._dataAllY, trainY])
                    initX = trainX.cpu().numpy()
                    initY = trainY.cpu().numpy()
                    for idx in range(initX.shape[0]): 
                        paretoParams, paretoValues = newParetoSet(paretoParams, paretoValues, list(initX[idx]), list(-initY[idx]))
                    
            t1 = time.time()
            if verbose:
                print(f"Batch {iter}: Pareto-front ->", paretoValues)
                print(f"[Hypervolume]:", f"(Batch {iter})", calcHypervolume(self._refpoint, paretoValues)) 
            else:
                print(".", end="")

        return list(zip(paretoParams, paretoValues))

        
    def optimizeLegacy(self, steps=2**4, regions=4, verbose=True): 
        self._states = []
        self._mlls   = []
        self._models = []
        self._dataXY = []
        self._dataAllX = None
        self._dataAllY = None
        # Initialize all regions
        initParams = []
        initValues = []
        for region in range(regions): 
            trainX, trainY = self.initSamples()
            state = MorboState(dim=self._nDims, batch_size=self._batchSize)
            initX = trainX.cpu().numpy()
            initY = trainY.cpu().numpy()
            for idx in range(initX.shape[0]): 
                initParams.append(list(initX[idx]))
                initValues.append(list(-initY[idx]))
            self._states.append(state)
            self._dataXY.append([trainX, trainY])
            if self._dataAllX is None: 
                self._dataAllX = trainX
            else: 
                self._dataAllX = torch.cat([self._dataAllX, trainX])
            if self._dataAllY is None: 
                self._dataAllY = trainY
            else: 
                self._dataAllY = torch.cat([self._dataAllY, trainY])
        paretoParams, paretoValues = pareto(initParams, initValues)
        print(f'[Initial PARETO]: {paretoParams}, {paretoValues}')
        print(f'[Hypervolume]:', calcHypervolume(self._refpoint, paretoValues))
        
        for iter in range(steps): 
            for region in range(regions): 
                with gpytorch.settings.max_cholesky_size(float("inf")):
                    newX, newY = self.getObservations(region)
                self._dataXY[region][0] = torch.cat([self._dataXY[region][0], newX])
                self._dataXY[region][1] = torch.cat([self._dataXY[region][1], newY])
                self._dataAllX = torch.cat([self._dataAllX, newX])
                self._dataAllY = torch.cat([self._dataAllY, newY])

                paretoPrev = str(paretoValues)
                tmpX = newX.cpu().numpy()
                tmpY = newY.cpu().numpy()
                for idx in range(tmpX.shape[0]): 
                    paretoParams, paretoValues = newParetoSet(paretoParams, paretoValues, list(tmpX[idx]), list(-tmpY[idx]))
                paretoPost = str(paretoValues)

                self._states[region].update(paretoPrev != paretoPost)
                regionHV = self.getRegionHV(region)
                print("[RegionHV]", regionHV)
                if self._states[region].restart_triggered or regionHV == 0.0: 
                    print(f"[Restart] region {region} restart")
                    trainX, trainY = self.initSamples()
                    self._states[region] = MorboState(dim=self._nDims, batch_size=self._batchSize)
                    self._dataXY[region][0] = trainX
                    self._dataXY[region][1] = trainY
                    self._dataAllX = torch.cat([self._dataAllX, trainX])
                    self._dataAllY = torch.cat([self._dataAllY, trainY])
                    initX = trainX.cpu().numpy()
                    initY = trainY.cpu().numpy()
                    for idx in range(initX.shape[0]): 
                        paretoParams, paretoValues = newParetoSet(paretoParams, paretoValues, list(initX[idx]), list(-initY[idx]))
                    
            t1 = time.time()
            if verbose:
                print(f"[Hypervolume]:", f"(Batch {iter})", calcHypervolume(self._refpoint, paretoValues)) 
            else:
                print(".", end="")

        return list(zip(paretoParams, paretoValues))

        
    def optSingle(self, steps=2**4, regions=4, verbose=True): 
        trainX, trainY = self.initSamples()
        state = MorboState(dim=self._nDims, batch_size=self._batchSize)

        initX = trainX.cpu().numpy()
        params = []
        values = []
        for param in list(initX): 
            params.append(list(param))
            values.append(self._evalPoint(param))
        paretoParams, paretoValues = pareto(params, values)
        print(f'[Initial PARETO]: {paretoParams}, {paretoValues}')
        
        for iter in range(steps):  
            mll, model = self.initModel(trainX, trainY)
            with gpytorch.settings.max_cholesky_size(float("inf")):
                fit_gpytorch_model(mll)
                newX, newY = self.getObservations(state, mll, model, trainX, trainY)
            trainX = torch.cat([trainX, newX])
            trainY = torch.cat([trainY, newY])

            paretoPrev = str(paretoValues)
            tmpBatchX = newX.cpu().numpy()
            for param in list(tmpBatchX): 
                paretoParams, paretoValues = newParetoSet(paretoParams, paretoValues, list(param), self._evalPoint(param))
            paretoPost = str(paretoValues)

            state.update(paretoPrev != paretoPost)
                
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
    parser.add_argument("-j", "--njobs", required=False, action="store", default=4)
    parser.add_argument("-m", "--morbos", required=False, action="store", default=4)
    return parser.parse_args()


if __name__ == "__main__": 

    # ref = [150.0, 150.0, 150.0]
    # values = [[101.39675846620108, 94.21187560837309, 93.20723787907446], [97.98392410067203, 97.23984705944494, 95.08606014753168], [99.22255896692582, 97.2741708503963, 92.95252506147153], [100.76426406641191, 94.49584999601551, 92.92604501607718], [97.28554486757149, 100.17752945634612, 94.46693146712062], [96.20503360126499, 100.17572055575462, 94.51232583065381], [96.74528923441824, 99.8419962256713, 94.51484773973898], [98.22110950059296, 100.02981982862399, 94.49467246705757], [99.4333904335222, 94.89272849108382, 93.44934115125149], [99.94729213335091, 94.6494547118283, 93.3610743332703], [97.36460666754513, 97.8056885564113, 95.29411764705883], [99.45974436684676, 95.08020425508265, 93.38629342412206], [98.80089603373303, 97.21400460362096, 95.08858205661686], [99.30162076689946, 95.08037311341079, 93.36611815144065], [100.52707866649098, 94.83609124533523, 93.26650274257614], [99.88140730003954, 95.38959136100549, 93.31315806065192], [97.06153643431283, 99.64279193948559, 95.62700964630226]] # morbo
    # values = [[99.88140730003954, 94.170592737822, 97.79837336864007], [99.98682303333773, 101.50797802688446, 93.07735956118782], [98.41876400052708, 94.35685727689214, 98.03038900447639], [100.03953089998683, 94.75406290621471, 97.76684950507536], [100.0527078666491, 100.79160439153121, 92.95882983418448], [99.47292133350902, 94.44183005435306, 97.76811045961793], [100.0527078666491, 94.53925118739153, 97.74415232330874], [99.47292133350902, 94.44183005435306, 97.76811045961793], [99.11714323362762, 94.34551455227513, 98.0278670953912], [99.34115166688629, 94.82422814260073, 94.73047096652166], [99.23573593358809, 95.10100741706755, 94.66490133030705], [99.10396626696534, 95.47570841819962, 95.2197213290461], [100.0527078666491, 100.79160439153121, 92.95882983418448], [100.0527078666491, 100.79160439153121, 92.95882983418448], [99.36750560021083, 93.96769473322132, 97.82989723220479], [99.11714323362762, 94.34551455227513, 98.0278670953912], [99.24891290025037, 95.05752386699744, 94.70525187566989], [99.16985110027672, 95.14550963835975, 94.65229178488116], [95.83607853472131, 110.32957050723202, 99.33673791059832], [99.36750560021083, 93.96769473322132, 97.82989723220479], [99.34115166688629, 94.82422814260073, 94.73047096652166], [99.40703650019765, 94.87551506391127, 94.62202887585903], [99.78916853340361, 94.85974774498216, 94.6245507849442], [99.80234550006588, 94.97791797773426, 94.59050501229432], [93.79364870206878, 111.1465917579836, 99.74402622785449], [98.41876400052708, 94.35685727689214, 98.03038900447639], [94.51838186849388, 109.49107089123518, 99.42500472857954], [98.41876400052708, 94.35685727689214, 98.03038900447639], [94.00448016866517, 110.0383510850827, 99.60027740999938], [99.80234550006588, 94.97791797773426, 94.59050501229432], [99.16985110027672, 95.08875046493554, 94.68759851207363], [99.16985110027672, 95.08875046493554, 94.68759851207363], [98.23428646725525, 94.56899279826132, 98.03417186810417], [97.75991566741337, 94.80297499847705, 97.9723850955173], [97.94439320068521, 94.59402408996404, 97.80089527772523], [97.19330610093556, 94.94910324725147, 97.9169030956434], [97.72038476742654, 94.91857807852658, 97.85385536851398], [99.10396626696534, 95.47570841819962, 95.2197213290461], [99.10396626696534, 95.47570841819962, 95.2197213290461], [99.10396626696534, 95.47570841819962, 95.2197213290461], [99.31479773356173, 94.57398355108238, 97.783241914129], [97.58861510080379, 94.92957434450302, 97.84755059580102], [97.19330610093556, 94.94910324725147, 97.9169030956434], [93.55646330214785, 111.35737848319764, 99.78437677321733], [93.54328633548556, 111.01003621205956, 99.80581300044133], [97.25919093424693, 95.96994961304298, 94.65355273942374], [98.41876400052708, 95.97392767599962, 94.57032973961289], [98.41876400052708, 95.45105257171694, 97.74541327785133], [98.73501120042167, 96.03385328815476, 94.54006683059076], [97.95757016734747, 95.48919338907224, 94.59050501229432], [98.53735670048755, 94.39745942292967, 97.84628964125842], [97.94439320068521, 94.59402408996404, 97.80089527772523], [97.95757016734747, 95.48919338907224, 94.59050501229432], [98.41876400052708, 95.97392767599962, 94.57032973961289], [98.73501120042167, 96.03385328815476, 94.54006683059076], [93.68823296877059, 110.9530055770179, 99.81085681861168], [93.68823296877059, 110.9530055770179, 99.81085681861168], [93.54328633548556, 111.01003621205956, 99.80581300044133], [95.66477796811175, 109.68582297684259, 99.41239518315365], [97.11424430096191, 102.98157336944969, 98.00516991362461], [94.21531163526156, 109.69571016107521, 99.46031145577203], [95.00592963499803, 110.07983341090694, 99.40356850135554], [94.51838186849388, 109.49107089123518, 99.42500472857954], [97.11424430096191, 102.98157336944969, 98.00516991362461], [97.11424430096191, 102.98157336944969, 98.00516991362461], [97.25919093424693, 95.96994961304298, 94.65355273942374], [97.25919093424693, 95.96994961304298, 94.65355273942374], [98.30017130056662, 98.67658987707667, 94.48710673980203], [98.30017130056662, 98.67658987707667, 94.48710673980203], [94.00448016866517, 110.0383510850827, 99.60027740999938], [94.00448016866517, 110.0383510850827, 99.60027740999938], [95.83607853472131, 110.32957050723202, 99.33673791059832], [94.72921333509026, 110.38359343957329, 99.35691318327974], [95.83607853472131, 110.32957050723202, 99.33673791059832], [94.21531163526156, 109.69571016107521, 99.46031145577203], [94.2021346685993, 109.87121929884134, 99.45652859214425], [97.11424430096191, 102.98157336944969, 98.00516991362461], [99.44656740018448, 95.6809499647468, 93.94489628648888], [99.26208986691263, 95.71406920228381, 93.95120105920182], [99.42021346685993, 95.91170660281153, 93.90958955929639], [99.47292133350902, 95.24893306382256, 93.86041233213543], [99.327974700224, 95.47837196278161, 93.94615724103146], [100.02635393332454, 95.08558218578686, 93.9297648319778], [99.14349716695217, 97.69078552526166, 94.05712124077927], [99.80234550006588, 95.87017688623034, 93.84275896853919], [99.92093820002636, 95.12385252254097, 93.87932665027427], [99.82869943339044, 95.17682204577524, 93.90454574112604], [96.52128080115956, 103.35180882697479, 95.67618687346322], [96.95612070101463, 103.276575080118, 95.83885000945716], [99.73646066675452, 101.61928580196847, 93.00296324317509], [100.02635393332454, 95.08558218578686, 93.9297648319778], [100.02635393332454, 95.08558218578686, 93.9297648319778], [99.3806825668731, 95.6837931801112, 93.93859151377593], [99.80234550006588, 95.87017688623034, 93.84275896853919], [99.86823033337726, 95.32795899144371, 93.83393228674107], [99.86823033337726, 95.32795899144371, 93.83393228674107], [99.80234550006588, 95.87017688623034, 93.84275896853919], [99.86823033337726, 95.32795899144371, 93.83393228674107], [99.80234550006588, 95.87017688623034, 93.84275896853919], [99.26208986691263, 95.71406920228381, 93.95120105920182], [99.26208986691263, 95.71406920228381, 93.95120105920182], [99.26208986691263, 95.71406920228381, 93.95120105920182], [99.26208986691263, 95.71406920228381, 93.95120105920182], [99.47292133350902, 95.24893306382256, 93.86041233213543], [99.47292133350902, 95.24893306382256, 93.86041233213543], [99.47292133350902, 95.24893306382256, 93.86041233213543], [96.52128080115956, 103.35180882697479, 95.67618687346322], [99.14349716695217, 97.69078552526166, 94.05712124077927], [99.14349716695217, 97.69078552526166, 94.05712124077927], [99.14349716695217, 97.69078552526166, 94.05712124077927], [99.14349716695217, 97.69078552526166, 94.05712124077927], [99.8945842667018, 102.30107292673719, 92.81634197087196], [99.8945842667018, 102.30107292673719, 92.81634197087196], [100.17130056660956, 102.53513081413271, 92.76968665279617], [96.95612070101463, 103.276575080118, 95.83885000945716], [96.95612070101463, 103.276575080118, 95.83885000945716]]
    # print(calcHypervolume(ref, values))

    # values = np.array(values)
    # print(100.0 - np.min(values, axis=0))

    # exit(1)
    
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
    model = Morbo(ndims, nobjs, names, types, ranges, funcEval, refpoint=refpoint, weights=[1.0/nobjs] * nobjs, nInit=ninit, batchSize=int(args.batchsize), nJobs=int(args.njobs))
    results = model.optimize(steps=steps, regions=int(args.morbos), verbose=True)
    print(results)
    list(map(lambda x: print("Parameter:", x[0], "\n -> Value:", x[1]), results))
    print("[Hypervolume]:", calcHypervolume([refpoint] * nobjs, list(map(lambda x: x[1], results))))

