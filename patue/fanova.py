import random
import multiprocessing as mp
from itertools import combinations

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ARDRegression
from sklearn.ensemble import RandomForestRegressor

from utils.utils import *

def getArea(split, ranges, indices=None): 
    assert ranges is None or len(ranges) == len(split)
    if indices is None: 
        indices = list(range(len(split)))
    if ranges is None: 
        ranges = [[0.0, 1.0] for _ in range(len(split))]
    split = np.array(split)
    ranges = np.array(ranges)
    assert split.shape[0] == ranges.shape[0] and split.shape[1] == ranges.shape[1]

    return np.prod((split[:, 1] - split[:, 0]) / (ranges[:, 1] - ranges[:, 0]))

def getSplits(tree, ranges): 
    splits = []
    areas = []
    preds = []

    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold
    value = tree.tree_.value
    node_depth = [0 for _ in range(n_nodes)]
    is_leaves = [False for _ in range(n_nodes)]
    is_split = [False for _ in range(n_nodes)]
    stack = [(0, 0, ranges.copy())]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth, split = stack.pop()
        node_depth[node_id] = depth
        # If the left and right child of a node is not the same we have a split node
        is_split_node = children_left[node_id] != children_right[node_id]
        if is_split_node:
            split_left = split.copy()
            split_right = split.copy()
            split_left[feature[node_id]][1] = threshold[node_id]
            split_right[feature[node_id]][0] = threshold[node_id]
            assert split_left[feature[node_id]][0] <= split_left[feature[node_id]][1]
            assert split_right[feature[node_id]][0] <= split_right[feature[node_id]][1]
            stack.append((children_left[node_id], depth + 1, split_left))
            stack.append((children_right[node_id], depth + 1, split_right))
            is_split[node_id] = True
        else:
            area = getArea(split, ranges)
            splits.append(split)
            areas.append(area)
            preds.append(value[node_id][0][0])
            assert len(value[node_id]) == 1 and len(value[node_id][0]) == 1
            is_leaves[node_id] = True

    return splits, areas, preds


def randomArray(ranges): 
    results = []
    for elem in ranges: 
        left, right = elem[0], elem[1]
        value = left + random.random() * (right - left)
        results.append(value)

    return results


def inRange(elem, split): 
    assert len(elem) == len(split)
    for idx in range(len(elem)): 
        if elem[idx] < split[idx][0] or elem[idx] > split[idx][1]: 
            return False
    return True



def fanovaTree(tree, ranges, K=2, SAMPLES=1024): 
    splits, areas, preds = getSplits(tree, ranges)
    f_empty = np.sum(np.array(areas) * np.array(preds))
    v = np.sum(np.array(areas) * (np.array(preds) - f_empty) * (np.array(preds) - f_empty))
    # list(map(lambda x: print(x), splits))

    def _calc_f_u(array, sample): 
        a_u = 0
        count = 0
        for jdx in range(len(splits)):  
            if inRange(sample, splits[jdx]): 
                tmp = ranges.copy()
                for kdx in array: 
                    tmp[kdx] = splits[jdx][kdx]
                area_i = getArea(tmp, ranges)
                a_u += area_i * preds[jdx]
                count += 1
        assert count == 1
                    
        f_u = a_u - f_empty
        todo = []
        k = len(array)
        if k > 1: 
            for jdx in range(1, k): 
                todo.extend(list(combinations(array, jdx)))
        for array2 in todo: 
            f_u_2 = _calc_f_u(array2, sample)
            f_u -= f_u_2

        return f_u

    results = {}
    for k in range(1, K+1): 
        for array in combinations(list(range(len(ranges))), k): 
            v_u = 0
            for idx in range(SAMPLES): # Monte Carlo
                sample = randomArray(ranges)
                f_u = _calc_f_u(array, sample)
                v_u += f_u * f_u
            v_u /= SAMPLES
            f_u = v_u / v
            results[array] = f_u
    return results
    

def _predict(model, ranges, X): 
    splits_array = []
    preds_array = []
    for tree in model.estimators_: 
        splits, areas, preds = getSplits(tree, ranges)
        splits_array.append(splits)
        preds_array.append(preds)

    Y = np.zeros([len(splits_array), X.shape[0]])
    assert len(splits_array) == len(preds_array)
    for idx in range(len(splits_array)): 
        for jdx, x in enumerate(X): 
            count = 0
            for kdx in range(len(splits_array[idx])): 
                if inRange(x, splits_array[idx][kdx]): 
                    Y[idx][jdx] = preds_array[idx][kdx]
                    count += 1
            assert count == 1
    Y = np.mean(Y, axis=0)

    return Y


def fanova(X, Y, ranges, K=2, SAMPLES=1024, 
           n_estimators=16, max_depth=8, min_samples_split=2, min_samples_leaf=1, \
           n_fanova_jobs=8): 
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, \
                                  min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, \
                                  random_state=0)
    model.fit(X, Y)
    predicted = model.predict(X)
    print("R^2 score:", model.score(X, Y))
    print("MSE1:", np.mean((predicted - Y) * (predicted - Y)))

    # predicted = _predict(model, ranges, X)
    # print("MSE2:", np.mean((predicted - Y) * (predicted - Y)))
    
    results = {}
    means = {}
    vars = {}
    pool = mp.Pool(processes=n_fanova_jobs)
    processes = []
    for tree in model.estimators_: 
        process = pool.apply_async(fanovaTree, (tree, ranges, K, SAMPLES))
        processes.append(process)
    pool.close()
    pool.join()
    for process in processes: 
        for key, value in process.get().items(): 
            if not key in results: 
                results[key] = []
            results[key].append(value)

    for key in results.keys(): 
        means[key] = np.mean(results[key])
        vars[key] = np.var(results[key])

    assert len(results) == len(means) and len(results) == len(vars)

    return results, means, vars


class FANOVA: 

    def __init__(self, nDims, nObjs, nameVars, typeVars, rangeVars, funcEval): 
        self._nDims      = nDims
        self._nObjs      = nObjs
        self._nameVars   = nameVars
        self._typeVars   = typeVars
        self._rangeVars  = rangeVars
        self._funcEval   = funcEval
        self._name2index = {}
        for idx, name in enumerate(self._nameVars): 
            self._name2index[name] = idx
        
        self._bound = [0.0, 1.0]
        self._bounds = [self._bound, ] * self._nDims

    def sample(self, nSamples=2**4): 
        samples = []
        for _ in range(nSamples):
            point = []
            for _ in range(len(self._nameVars)): 
                point.append(random.random())
            samples.append(point)

        return samples

    def fanova(self, nSamples=2**4, nMC=1024, n_estimators=64, max_depth=16, n_fanova_jobs=8): 
        def objective(variables): 
            values = []
            for idx, name in enumerate(self._nameVars): 
                typename = self._typeVars[idx]
                value = None
                if typename == "int": 
                    value = round(variables[idx] * (self._rangeVars[idx][1] - self._rangeVars[idx][0])) + self._rangeVars[idx][0]
                elif typename == "float": 
                    value = variables[idx] * (self._rangeVars[idx][1] - self._rangeVars[idx][0]) + self._rangeVars[idx][0]
                elif typename == "enum": 
                    value = self._rangeVars[idx][round(variables[idx] * (len(self._rangeVars[idx]) - 1))]
                else: 
                    assert typename in ["int", "float", "enum"]
                assert not value is None
                values.append(value)
            score = self._funcEval(dict(zip(self._nameVars, values)))
            print(f'[Evaluation SCORE]: {score}')
            return score
        
        samples = self.sample(nSamples)
        objs = []
        for point in samples: 
            objs.append(objective(point))

        X = np.array(samples)
        Ys = np.array(objs)
        results = [] 
        means = [] 
        vars = []
        for idx in range(Ys.shape[1]): 
            Y = Ys[:, idx]
            ranges = np.array(self._bounds)
            result, mean, var = fanova(X, Y, ranges, K=1, SAMPLES=nMC, 
                                       n_estimators=n_estimators, max_depth=max_depth, \
                                       min_samples_split=2, min_samples_leaf=1, \
                                       n_fanova_jobs=n_fanova_jobs)
            results.append(result)
            means.append(mean)
            vars.append(var)
        
        return results, means, vars


def trivial(): 
    from sklearn.datasets import make_regression
    n_features = 16
    n_informative = 8
    ranges = np.array([[-5.0, 5.0] for _ in range(n_features)])
    X, Y = make_regression(n_samples=1024, n_features=n_features, n_informative=n_informative, 
                           random_state=0, shuffle=False)
    reg = LinearRegression()
    reg.fit(X, Y)
    predicted = reg.predict(X)
    print("Linear coefficients:", reg.coef_)
    print("Score:", reg.score)
    print("MSE:", np.mean((predicted - Y) * (predicted - Y)))
    
    results, means, vars = fanova(X, Y, ranges, K=1, SAMPLES=1024, 
                                  n_estimators=64, max_depth=16, \
                                  min_samples_split=2, min_samples_leaf=1, \
                                  n_fanova_jobs=8)
    for key in results.keys(): 
        print("Key:", key, "; Mean:", means[key], "; Var:", vars[key])

from analytic_funcs import *
def test0(): 
    nDims = 16
    nObjs = 2 
    nameVars = ["x" + str(idx) for idx in range(nDims)] 
    typeVars = ["float" for idx in range(nDims)]
    rangeVars = [[0.0, 1.0] for idx in range(nDims)]
    funcEval = lambda x: [Powell(list(x.values()), nDims), Perm(list(x.values()), nDims)]
    model = FANOVA(nDims, nObjs, nameVars, typeVars, rangeVars, funcEval)
    # results, means, vars = model.fanova(nSamples=2**4, nMC=64, n_estimators=16, max_depth=8, n_fanova_jobs=8)
    results, means, vars = model.fanova(nSamples=2**10, nMC=1024, n_estimators=64, max_depth=16, n_fanova_jobs=8)
    for idx in range(len(results)): 
        print("Objective", idx)
        for key in results[idx].keys(): 
            print(" -> Key:", key, "; Mean:", means[idx][key], "; Var:", vars[idx][key])

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
    model = FANOVA(ndims, nobjs, names, types, ranges, funcEval)
    results, means, vars = model.fanova(nSamples=steps, nMC=1024, n_estimators=64, max_depth=16, n_fanova_jobs=8)
    for idx in range(len(results)): 
        print("Objective", idx)
        for key in results[idx].keys(): 
            print(" -> Key:", key, "; Mean:", means[idx][key], "; Var:", vars[idx][key])

