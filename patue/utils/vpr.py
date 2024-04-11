import os
import re
import time
import multiprocessing as mp

def regex_task(r_type):
    if r_type == "circuit":
        return (re.compile("circuit_list_add=(.*)"))
    elif r_type == "arch":
        return (re.compile("arch_list_add=(.*)"))
    elif r_type == "vpr_params":
        return (re.compile("vpr_params=(.*)"))
    elif r_type == "circuits_dir":
        return (re.compile("circuits_dir=(.*)"))
    elif r_type == "archs_dir":
        return (re.compile("archs_dir=(.*)"))
    elif r_type == "status":
        return (re.compile("VPR suceeded"))
    elif r_type == "pack":
        return (re.compile(" --pack "))
    elif r_type == "place":
        return (re.compile(" --place "))
    elif r_type == "route":
        return (re.compile(" --route "))
    elif r_type == "route_chan_width":
        return (re.compile(" --route_chan_width "))
    elif r_type == "compare_pair":
        return (re.compile("compare_pair_add=(.*),(.*)"))
    elif r_type == "analyze_single":
        return (re.compile("analyze_add=(.*),(.*),(.*)"))
    elif r_type == "analyze_all":
        return (re.compile("analyze_add=(.*),(.*)"))
    elif r_type == "optimize_arch":
        return (re.compile("optimize_arch_add=(.*)"))
    elif r_type == "is_mutable_chan_width":
        return (re.compile("is_mutable_chan_width=true"))

def regex_reporter(r_type):
    if r_type == "logic_area":
        return re.compile(".*?Total logic block area \(Warning, need to add pitch of routing to blocks with height > 3\): (.*)")
    elif r_type == "routing_area":
        return re.compile("	Total routing area: (.*?), per logic tile")
    elif r_type == "delay":
        return re.compile("Final critical path: (.*?) ns")
    elif r_type == "chan_width":
        return re.compile("Circuit successfully routed with a channel width factor of ([0-9]+).")
    elif r_type == "routable":
        return re.compile("Circuit is unroutable with a channel width factor")
    elif r_type == "max_channel":
        return re.compile("\s+([0-9]+)\s+([0-9]+)\s+(.*?)\s+300")
    elif r_type == "channel":
        return re.compile("Best routing used a channel width factor of ([0-9]+).")
    elif r_type == "routing_area_bidir":
        return re.compile(".*?Assuming no buffer sharing \(pessimistic\). Total: (.*?), per logic tile: (.*?)")

def report(folder): 
    with open(folder + "/vpr.out") as f: 
        area_found = False
        logic_area_found = False
        delay_found = False
        routable = True
        route_sucess = False
        lines = f.readlines()
        for line in lines:
            area_match = re.match(regex_reporter("routing_area"), line)
            if area_match != None:
                routing_area = float(area_match.group(1))
                area_found = True
            logic_area_match = re.match(regex_reporter("logic_area"), line)
            if logic_area_match != None:
                logic_area = float(logic_area_match.group(1))
                logic_area_found = True
            delay_match = re.match(regex_reporter("delay"), line)
            if delay_match != None:
                delay = float(delay_match.group(1))
                delay_found = True
            routable_match = re.match(regex_reporter("routable"), line)
            if routable_match != None:
                routable = False
            routable_match2 = re.match(regex_task("status"), line)
            if routable_match2 != None:
                route_sucess = True
    
    if area_found == False or delay_found == False or logic_area_found == False:
        return "ERR", "ERR"
    
    return routing_area, delay

# BENCHES = ["and_latch", "arm_core", "bgm", "blob_merge", "boundtop", "ch_intrinsics", \
#            "diffeq1", "diffeq2", "LU32PEEng", "LU64PEEng", "LU8PEEng", "mcml", \
#            "mkDelayWorker32B", "mkPktMerge", "mkSMAdapter4B", "or1200", "raygentop", \
#            "sha", "spree", "stereovision0", "stereovision1", "stereovision2", "stereovision3"]
# BENCHES = ["and_latch", "ch_intrinsics", "diffeq2", "stereovision3"]
# BASELINES = [(26067.2, 0.569618, 0.5392575263977051), (1149460.0, 2.0692, 3.7753467559814453), (1837750.0, 15.2426, 4.9320902824401855), (379107.0, 1.95837, 1.699131727218628)] # width=150
BENCHES = ["and_latch", "ch_intrinsics", "diffeq2", "mkPktMerge", "mkSMAdapter4B", "or1200", "raygentop", "stereovision3"]
BASELINES = [(21074.0, 0.569618, 0.48897242546081543), (944356.0, 2.1894, 5.556310176849365), (1510850.0, 15.8558, 7.975632190704346), (5011580.0, 4.13597, 21.34809947013855), (2459780.0, 4.35985, 23.97220230102539), (4700060.0, 11.2083, 36.43233132362366), (3036350.0, 4.31667, 34.227802753448486), (310589.0, 2.07857, 2.323246955871582)] # width=120

def runCommand(dir, cmd): 
    beginTime = time.time()
    pwd = os.getcwd()
    if not os.path.exists(dir):
        os.system("mkdir " + dir)
    # else: 
    #     os.system("rm -rf " + dir)
    os.chdir(dir)
    os.system(cmd)
    os.chdir(pwd)
    endTime = time.time()
    return endTime - beginTime

def runVPR(workDir, config={}, arg="", njobs=4, benches=BENCHES, arch="k6_N8_gate_boost_0.2V_22nm"): 
    if not os.path.exists(workDir):
        os.system("mkdir " + workDir)
    processes = []
    pool = mp.Pool(processes = njobs)
    for bench in benches:
        tmpdir = workDir + "/" + bench
        options = " " + arg
        for key, value in config.items(): 
            options += " --" + key + " " + str(value) 
        cmd = f"$VTR_ROOT/vpr/vpr $VTR_ROOT/vtr_flow/arch/timing/{arch}.xml $VTR_ROOT/prepared/{bench}/{bench}.pre-vpr.blif {options} > vpr.out 2>&1"
        processes.append(pool.apply_async(func=runCommand, args=(tmpdir, cmd, )))
    pool.close()
    pool.join()
    
    failed = False
    results = []
    for idx, bench in enumerate(benches):
        tmpdir = workDir + "/" + bench
        tmp = list(report(tmpdir))
        tmp.append(processes[idx].get())
        results.append(report(tmpdir) + (processes[idx].get(), ))
        if "ERR" in results[-1]: 
            failed = True

    if failed: 
        return None
    return results

import sys
import numpy as np
def getBaseline(): 
    arg = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else ""
    results = runVPR("run/tmp", arg=arg, njobs=4, benches=BENCHES, arch="k6_N8_gate_boost_0.2V_22nm")
    print("runVPR:", results)
if __name__ == "__main__": 
    # getBaseline()
    # exit(1) 
    args = sys.argv[1:]
    for idx in range(len(args)): 
        if args[idx] == "--route_chan_width": 
            args[idx+1] = str(round(int(args[idx+1]) / 2) * 2)
    arg = " ".join(args) if len(sys.argv) > 1 else ""
    results = runVPR("run/tmp", arg=arg, njobs=4, benches=BENCHES, arch="k6_N8_gate_boost_0.2V_22nm")
    if results is None: 
        print("ERR ERR ERR")
    else: 
        ratios = []
        for idx, metrics in enumerate(BASELINES): 
            ratios.append([100.0 + 100.0 * (results[idx][0] - metrics[0]) / metrics[0], \
                           100.0 + 100.0 * (results[idx][1] - metrics[1]) / metrics[1], \
                           100.0 + 100.0 * (results[idx][2] - metrics[2]) / metrics[2]])
        means = np.mean(np.array(ratios), axis=0)
        # routing_area, delay, runtime
        print(means[0], means[1], means[2])