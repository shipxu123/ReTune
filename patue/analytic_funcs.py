import os
import sys 
sys.path.append(".")

import math
import numpy as np
from copy import deepcopy


def Currin(x, d=2):
    return -1*float(((1 - math.exp(-0.5*(1/x[1]))) * ((2300*pow(x[0],3) + 1900*x[0]*x[0] + 2092*x[0] + 60)/(100*pow(x[0],3) + 500*x[0]*x[0] + 4*x[0] + 20))))


def branin(x1,d=2):
    x=deepcopy(x1)
    x[0]= 15* x[0]-5
    x[1]=15*x[1]
    return -1*float(np.square(x[1] - (5.1/(4*np.square(math.pi)))*np.square(x[0]) + (5/math.pi)*x[0]- 6) + 10*(1-(1./(8*math.pi)))*np.cos(x[0]) + 10)


def Powell(xx,d):
    vmin=-4
    vmax=5
    x=[None]+list(vmin + np.asarray(xx) * (vmax-vmin))
    f_original=0
    for i in range(1,int(math.floor(d/4)+1)):
        f_original=f_original+pow(x[4*i-3]+10*x[4*i-2],2)+5*pow(x[4*i-1]-x[4*i],2)+pow(x[4*i-2]-2*x[4*i-1],4)+10*pow(x[4*i-3]-2*x[4*i],4)
    return -1*float(f_original)


def Perm(xx,d):
    vmin=-1*d
    vmax=d
    beta=10
    x=[None]+list(vmin + np.asarray(xx) * (vmax-vmin))
    f_original=0
    for i in range(1,d+1):
        sum1=0
        for j in range(1,d+1):
            sum1=sum1+(j+beta)*(x[j]-math.pow(j,-1*i))        
        f_original=f_original+math.pow(sum1,2)
    return -1*f_original


def Dixon(xx,d):
    vmin=-10
    vmax=10
    x=[None]+list(vmin + np.asarray(xx) * (vmax-vmin))
    f_original=0
    for i in range(2,d+1):    
        f_original=f_original+i*math.pow(2*math.pow(x[i],2)-x[i-1],2)
    f_original=f_original+math.pow(x[1]-1,1)
    return -1*f_original


def ZAKHAROV(xx,d):
    vmin=-5
    vmax=10
    x=[None]+list(vmin + np.asarray(xx) * (vmax-vmin))
    term1=0
    term2=0
    for i in range(1,d+1):
        term1=term1+x[i]**2
        term2=term2+0.5*i*x[i]
    f_original=term1+math.pow(term2,2)+math.pow(term2,4)
    return -1*f_original


def RASTRIGIN(xx,d):
    vmin=-5.12
    vmax=5.12
    x=[None]+list(vmin + np.asarray(xx) * (vmax-vmin))
    f_original=0
    for i in range(1,d+1):
        f_original=f_original+(x[i]**2-10*math.cos(2*x[i]*math.pi))
    f_original=f_original+10*d
    return -1*f_original


def SumSquares(xx,d):
    vmin=-5.12
    vmax=5.12
    x=[None]+list(vmin + np.asarray(xx) * (vmax-vmin))
    f_original=0
    for i in range(1,d+1):
        f_original=f_original+(i*math.pow(x[i],2))
    return -1*f_original

import argparse
def parseArgs(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, action="store")
    parser.add_argument("--b", required=True, action="store")
    parser.add_argument("--c", required=True, action="store")
    parser.add_argument("--d", required=True, action="store")
    return parser.parse_args()

if __name__ == "__main__": 
    args = parseArgs()
    x = [float(args.a), float(args.b), float(args.c), float(args.d)]
    print(str(Powell(x, 4)), str(Perm(x, 4)))
