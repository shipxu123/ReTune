import random

import numpy as np
import numpy_indexed as npi
import pandas as pd
from sklearn import preprocessing
from xgboost import XGBRegressor

from utils.utils import *

class FIST: # FIST only supports enum values
    
    FLOAT_STEPS = 4

    def __init__(self, nDims, nObjs, nameVars, typeVars, rangeVars, funcEval, nInit=2**4): 
        assert len(nameVars) == len(rangeVars)
        self._funcEval = funcEval
        self._ndim     = nDims
        self._nobjs    = nObjs
        self._names    = nameVars
        self._types    = typeVars
        self._values   = [[] for _ in range(len(nameVars))]
        for idx, typename in enumerate(self._types): 
            if typename == "int": 
                step = 1
                if rangeVars[idx][1]+1 - rangeVars[idx][0] > FIST.FLOAT_STEPS: 
                    step = int(round((rangeVars[idx][1]+1 - rangeVars[idx][0]) / FIST.FLOAT_STEPS))
                self._values[idx].extend(list(range(rangeVars[idx][0], rangeVars[idx][1]+1, step)))
            elif typename == "float": 
                step = (rangeVars[idx][1] - rangeVars[idx][0]) / FIST.FLOAT_STEPS
                if step == 0.0: 
                    self._values[idx].append(rangeVars[idx][0])
                else: 
                    point = rangeVars[idx][0]
                    while point < rangeVars[idx][1]: 
                        self._values[idx].append(point)
                        point += step
            else: 
                self._values[idx].extend(list(range(len(rangeVars[idx]))))
        self._rangeVars = rangeVars
        self._nInit    = nInit
        self._pointsSampled  = []
        self._resultsSampled = []
        self._pointsApprox   = {} 
        self._resultsApprox  = {} 


    def _randomSampleOne(self, fixedIndices=[], fixedValues=[]): 
        values = []
        for idx in range(self._ndim): 
            value = self._values[idx][random.randint(0, len(self._values[idx]) - 1)]
            values.append(value)
        for idx, index in enumerate(fixedIndices): 
            values[index] = fixedValues[idx]

        return values


    def _randomSampleAll(self, nAtLeast=None): 
        points = []
        if not nAtLeast is None: 
            used = [set() for _ in range(len(self._names))]
            okay = False
            while (not okay) or (len(points) < nAtLeast): 
                sample = self._randomSampleOne()
                for idx, feature in enumerate(sample): 
                    used[idx].add(feature)
                okay = True
                for idx, pointsUsed in enumerate(used): 
                    if len(pointsUsed) < len(self._values[idx]): 
                        okay = False
                points.append(sample)
        else: 
            indices = [0 for _ in range(len(self._names))]
            okay = False
            while not okay: 
                sample = [self._values[idx][index] for idx, index in enumerate(indices)]
                points.append(sample)
                indices[-1] += 1
                for _idx in range(len(indices)): 
                    idx = len(indices) - 1 - _idx
                    assert indices[idx] <= len(self._values[idx])
                    if indices[idx] == len(self._values[idx]): 
                        if idx == 0: 
                            okay = True
                        else: 
                            indices[idx] = 0
                            indices[idx-1] += 1

        return points


    def _efficient(self, costs):
        efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if efficient[i]:
                efficient[efficient] = np.any(costs[efficient]<=c, axis=1)
        return efficient


    def ADRS(self, predicted, labels, scale): 
        efficient = self._efficient(labels)
        selected = labels[efficient]
        unified_labels = np.unique(selected, axis=0)
        
        efficient = self._efficient(predicted)
        selected = predicted[efficient]
        unified_predicted = np.unique(selected, axis=0)
        
        results = []
        for label in unified_labels:
            results.append(np.amax((unified_predicted - label)/scale, axis=1).min())
        return sum(results) / len(results)

    
    def _evalPoints(self, points): 
        results = []
        for point in points: 
            if point in self._pointsSampled: 
                results.append(self._resultsSampled[self._pointsSampled.index(point)])
            else: 
                config = dict(zip(self._names, point))
                for idx, name in enumerate(self._names): 
                    if self._types[idx] == "enum": 
                        config[name] = self._rangeVars[idx][int(config[name])]
                    elif self._types[idx] == "int": 
                        config[name] = int(config[name])
                results.append(self._funcEval(config))
            print("[Eval] Point:", config)
            print(" -> Result:", results[-1])
            with open("historyFIST.txt", "a+") as fout: 
                fout.write(str(config) + "\n")
                fout.write(str(results[-1]) + "\n")
        return results


    def learn(self, X_train, Y_train, X_test, n_labels=None, depth=12, seed=0): 
        if n_labels is None: 
            n_labels = Y_train.shape[1]
        assert n_labels <= Y_train.shape[1]

        predicted = []
        for idx in range(n_labels): 
            model = XGBRegressor(max_depth=depth, random_state=seed, verbosity=0).fit(X_train, Y_train[:, idx])
            predicted.append(model.predict(X_test))
        predicted = np.column_stack(predicted)

        efficient = self._efficient(predicted)

        return predicted, efficient

    def _importance(self, features, labels, ndim=None): 
        features = np.array(features)
        labels = np.array(labels)
        if ndim is None: 
            ndim = features.shape[1]

        results = []
        for idx in range(ndim): 
            selected = []
            for jdx in range(ndim): 
                if idx == jdx: 
                    continue
                selected.append(jdx)
            selected = features[:, selected]
            grouped = npi.group_by(selected).split(np.arange(len(selected)))
            vars = []
            for group in grouped: 
                vars.append(np.var(labels[group]))
            results.append(np.sum(vars))

        return results


    def _importantIndices(self, features, labels, ndim=None, proportion=0.5): 
        features = np.array(features)
        labels = np.array(labels)
        variance = self._importance(features, labels, ndim)
        indices = np.argsort(list(-np.array(variance)))
        indices = indices[0:int(len(indices)*proportion)]
        selected = features[:, indices]
        grouped = npi.group_by(selected).split(np.arange(selected.shape[0]))
        print("Variance:", variance)

        return indices, grouped


    def modelLess(self, featuresInit, impIndices, impGrouped, nPoints=16): 
        featuresInit = np.array(featuresInit)
        points = []
        indicesGrouped = list(range(len(impGrouped)))
        random.shuffle(indicesGrouped)
        print("Model-less Points", min(nPoints, len(indicesGrouped)))
        for idx in range(min(nPoints, len(indicesGrouped))): 
            index = impGrouped[indicesGrouped[idx]][0]
            fixed = featuresInit[index][impIndices]
            point = self._randomSampleOne(impIndices, fixed)
            points.append(point)
        results = self._evalPoints(points)
        self._pointsSampled.extend(points)
        self._resultsSampled.extend(results)
        for idx, point in enumerate(points): 
            key = str(list(np.array(point)[impIndices]))
            self._pointsApprox[key] = point
            self._resultsApprox[key] = results[idx]


    def modelGuided(self, nExplore, nPoints, impIndices, nMC=1024, seed=0): 
        samplesAll = None
        if len(self._names) < 16: 
            samplesAll = self._randomSampleAll()
        for idx in range(nPoints): 
            if len(self._names) >= 16: 
                samplesAll = self._randomSampleAll(nMC)
            depth = 6 + min(int(10 * 1.0 * idx / nPoints), 7)
            if idx < nExplore: 
                X_train = []
                Y_train = []
                X_test = []
                for key, point in self._pointsApprox.items(): 
                    result = self._resultsApprox[key]
                    X_train.append(point)
                    Y_train.append(result)
                used = set()
                for point in X_train: 
                    used.add(str(point))
                for point in samplesAll: 
                    if not str(point) in used: 
                        X_test.append(point)
                X_train = np.array(X_train)
                Y_train = np.array(Y_train)
                if len(X_test) == 0: 
                    continue
                X_test  = np.array(X_test)
                Y_test, pareto = self.learn(X_train, Y_train, X_test, depth=depth, seed=seed)
                indices = []
                for jdx, sign in enumerate(pareto): 
                    if sign: 
                        indices.append(jdx)
                index = random.choice(indices)
                key = str(list(X_test[index][impIndices]))
                points = [list(X_test[index]), ]
                results = self._evalPoints(points)
                self._pointsSampled.extend(points)
                self._resultsSampled.extend(results)
                self._pointsApprox[key] = list(X_test[index])
                self._resultsApprox[key] = list(Y_test[index])
            else: 
                X_train = []
                Y_train = []
                X_test = []
                for jdx, point in enumerate(self._pointsSampled): 
                    result = self._resultsSampled[jdx]
                    X_train.append(point)
                    Y_train.append(result)
                used = set()
                for point in X_train: 
                    used.add(str(point))
                for point in samplesAll: 
                    if not str(point) in used: 
                        X_test.append(point)
                X_train = np.array(X_train)
                Y_train = np.array(Y_train)
                if len(X_test) == 0: 
                    continue
                X_test  = np.array(X_test)
                Y_test, pareto = self.learn(X_train, Y_train, X_test, depth=depth, seed=seed)
                indices = []
                for jdx, sign in enumerate(pareto): 
                    if sign: 
                        indices.append(jdx)
                index = random.choice(indices)
                points = [list(X_test[index]), ]
                results = self._evalPoints(points)
                self._pointsSampled.extend(points)
                self._resultsSampled.extend(results)
        pareto = self._efficient(np.array(self._resultsSampled))
        indices = []
        for idx, sign in enumerate(pareto): 
            if sign: 
                indices.append(idx)
        pointsPareto = []
        resultsPareto = []
        for index in indices: 
            pointsPareto.append(self._pointsSampled[index])
            resultsPareto.append(self._resultsSampled[index])
        return pointsPareto, resultsPareto

    def optimize(self, nInitAtLeast=256, nModelLess=32, nExplore=64, nPoints=1024, nMC=1024): 
        pointsInit = self._randomSampleAll(nInitAtLeast)
        print("[Init]:", len(pointsInit), "points;")
        resultsInit = self._evalPoints(pointsInit)
        self._pointsSampled.extend(pointsInit)
        self._resultsSampled.extend(resultsInit)
        impIndices, impGrouped = self._importantIndices(pointsInit, resultsInit)
        self.modelLess(pointsInit, impIndices, impGrouped, nPoints=nModelLess)
        print("[Model-less]:", len(self._pointsApprox), "points;")
        pointsPareto, resultsPareto = self.modelGuided(nExplore, nPoints, impIndices, nMC=nMC, seed=0)
        print("[Model-guided]:", nPoints, "points;")
        print(" -> Pareto optimal:", resultsPareto)
        return list(zip(pointsPareto, resultsPareto))


import argparse
def parseArgs(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--paramfile", required=True, action="store")
    parser.add_argument("-c", "--command", required=True, action="store")
    parser.add_argument("-r", "--refpoint", required=True, action="store")
    parser.add_argument("-n", "--nobjs", required=True, action="store")
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
    ndims = len(names)
    model = FIST(ndims, nobjs, names, types, ranges, funcEval, nInit=ninit)
    results = model.optimize(nInitAtLeast=ninit, nModelLess=ninit, nExplore=ninit, nPoints=steps, nMC=1024)
    list(map(lambda x: print("Parameter:", x[0], "\n -> Value:", x[1]), results))
    print("[Hypervolume]:", calcHypervolume([refpoint] * nobjs, list(map(lambda x: x[1], results))))
