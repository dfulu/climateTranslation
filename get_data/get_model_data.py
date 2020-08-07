"""
This python script is used to create a shell scrpt to download data from 
the NERSC portal. The scripts created can download models from a given set and can download the 
temperature at surface only from the climate of the 20th century (C20C) experiment.

References
----------
[1] https://www.wcrp-climate.org/modelling-wgcm-mip-catalogue/modelling-wgcm-mips/245-modelling-wgcm-catalogue-c20c
"""

import re
import argparse
import os
dirname = os.path.dirname(__file__)

models = ['CAM5', 'MIROC5','HadGEM3']
variables = ['tas', 'tasmax', 'tasmin']
experiments = ['all_hist', 'nat_hist']

def check_model(model):
    model = str(model)
    if model not in models:
        raise argparse.ArgumentTypeError("{} model cannot be used here. Must be {}".format(model, models))
    return model

def check_variable(st):
    st = str(st)
    if st not in variables:
        raise argparse.ArgumentTypeError("variable {} not in {}.".format(st, variables))
    return st

def check_experiment(experiment):
    experiment = str(experiment)
    if experiment not in experiments:
        raise argparse.ArgumentTypeError("experiment {} not in {}.".format(experiment, experiments))
    return experiment

def check_start(value):
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError("%s is a valid start" % value)
    return ivalue

def check_stop(value):
    ivalue = int(value)
    if ivalue > 30:
        raise argparse.ArgumentTypeError("%s is a valid stop" % value)
    return ivalue

parser = argparse.ArgumentParser(description='Generate C20C download scripts.')
parser.add_argument('model', type=check_model, help="Model name from : {}.".format(models))
parser.add_argument('--variable', type=check_variable, help="Variable from : {} [tas]".format(variables), default='tas')
parser.add_argument('--experiment', type=check_experiment, help="Experiment from : {} [nat_hist]".format(experiments), default='nat_hist')
parser.add_argument('--start', type=check_start, help='run number to start at (inclusive) [1]', default=1)
parser.add_argument('--stop', type=check_stop, help='run number to stop at (inclusive) [150]', default=150)
parser.add_argument('--outputfile', type=str, help='output filename', default='')
args = parser.parse_args()



runstrings = {
    'CAM5':'run{:03d}',
    'MIROC5':'run{:03d}',
    'HadGEM3':'r1i1p{}', # stochastic phyics I think
}

runforms = {
    'CAM5':'run[0-9][0-9][0-9]',
    'MIROC5':'run[0-9][0-9][0-9]',
    'HadGEM3':'r1i1p[0-9]+', 
}

# check if line is a wget line
rexp1 = re.compile(runforms[args.model])
# check if in range of runs
rexp2 = re.compile('('+'|'.join([runstrings[args.model].format(i) for i in range(args.start, args.stop+1)]) + ')')
rexp3 = re.compile('/{}/'.format(args.variable))

prefix = 'runs{:02d}-{:02d}'.format(args.start, args.stop)

if args.outputfile=='':
    args.outputfile = "{}_{}_{}_{}.sh".format(args.model, args.experiment, args.variable, prefix)

def filter(s):
    return (not bool(rexp1.search(s))) or (bool(rexp2.search(s)) and bool(rexp3.search(s)))

def dedup_print(L):
    nL=[L[0]]
    for l in L[1:]:
        if l!=nL[-1]:
            nL.append(l)
    return nL


filename = os.path.join(dirname, "wget_scripts", "{}_{}.sh".format(args.model.lower(), args.experiment))
with open (filename, "r") as myfile:
    data = myfile.readlines()
data = [d[:-1] for d in data if filter(d)]
with open(args.outputfile, 'w') as f:
    for item in data:
        f.write("%s\n" % item)
