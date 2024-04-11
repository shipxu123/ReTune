import os
import sys 
sys.path.append(".")

from os.path import abspath
from datetime import datetime
from multiprocessing import cpu_count
from subprocess import run

from hyperopt import hp, fmin, tpe

from utils.utils import *

class TPE: 

    def __init__(self, nDims, nObjs, nameVars, typeVars, rangeVars, funcEval, weights, refpoint=0.0, nInit=2**4): 
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
        self._space = []
        for idx, name in enumerate(self._nameVars): 
            self._space.append(hp.uniform(name, 0.0, 1.0))
        self._visited = {}
        self._paretoParams = []
        self._paretoValues = []
        
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
        self._paretoParams, self._paretoValues = newParetoSet(self._paretoParams, self._paretoValues, values, score)
        print("[Eval] Point:", config)
        print(" -> Result:", score)
        print(" -> Pareto-front:", self._paretoValues)
        print(f'[Hypervolume]:', calcHypervolume(self._refpoint, self._paretoValues))
        with open("historyTPE.txt", "a+") as fout: 
            fout.write(str(config) + "\n")
            fout.write(str(score) + "\n")
        return score
        
    def optimize(self, steps=2**4): 
        def objective(variables): 
            score = self.objective(variables)
            return np.sum(np.array(score) * np.array(self._weights))

        bestX = fmin(objective, self._space, algo=tpe.suggest, max_evals=steps)
        bestY = objective(list(bestX.values()))

        # return [[bestX, bestY], ]
        return list(zip(self._paretoParams, self._paretoValues))


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
    model = TPE(ndims, nobjs, names, types, ranges, funcEval, refpoint=refpoint, weights=[1.0/nobjs] * nobjs, nInit=ninit)
    results = model.optimize(steps=steps)
    list(map(lambda x: print("Parameter:", x[0], "\n -> Value:", x[1]), results))
    print("[Hypervolume]:", calcHypervolume([refpoint] * nobjs, list(map(lambda x: x[1], results))))
    
