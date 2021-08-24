import xarray as xr
import zarr
import numpy as np

import argparse
import time
import os
import re

from dask.distributed import Client
from dask.diagnostics import ProgressBar

import progressbar

from climatetranslation.unit.data import construct_regridders, split_lon_at
from climatetranslation.unit.utils import get_config

print(f"staring - {time.asctime()}", flush=True)

###############################################
# input args
###############################################

class Filepathcheck:
    
    def inputzarr(self, filepaths):
        if len(filepaths) not in [1,2]:
            raise ValueError(f"inputzarr : {filepaths} must be either 1 or 2 files.")
        self.n_inputs = len(filepaths)
        return
    
    def outputzarr(self, filepaths):
        if len(filepaths) != self.n_inputs:
            raise ValueError(f"outputzarr : {filepaths} must have same number of files as input zarr list ({self.n_inputs}).")
        self.outputzarrs = filepaths
        return
    
    def intermediatezarr(self, filepaths):
        if filepaths != '':
            if len(filepaths) != self.n_inputs:
                raise ValueError(f"intermediatezarr : {filepaths} must have same number of files as input zarr list  ({self.n_inputs}).")
        else:
            filepaths = [f + '_intermediate' for f in self.outputzarrs]
        return filepaths
    
    def config(self, configpath):
        if self.n_inputs==1 and configpath!='':
            raise ValueError(
                "Cannot apply regridding with only one dataset. Do not supply regridding config if only 1 dataset"
            ) 
        if configpath!='':
            conf = get_config(configpath)
        else:
            conf = None
        return conf

    
parser = argparse.ArgumentParser()
parser.add_argument('--inputzarr', 
                    type=str, 
                    nargs='+', 
                    help='Input filename(s) - either 1 or 2 zarr files'
)
parser.add_argument('--outputzarr', 
                    type=str, 
                    nargs='+', 
                    help='Output filename(s) - either 1 or 2 zarr files, must be same number as input'
)
parser.add_argument('--intermediatezarr', 
                    type=str, 
                    help=('Rechunking intermediate store filename(s) - either 1 or 2 zarr files, must be same number as input.'
                        + 'Defaults to [inputzarr]_intermediate'), 
                    default=''
)

parser.add_argument('--latlonchunks', type=int, help='New chunk size for both lat and lon', default=4)
parser.add_argument('--max_mem', type=lambda x: f"{x}GB", help='Max RAM usgae in GB', default='40')
parser.add_argument('--n_workers', type=int, help='Number of workers', default=1)
parser.add_argument('--config', type=str, help='Path to the config file for regridding.', default='')
args = parser.parse_args()

filepathcheck = Filepathcheck()
filepathcheck.inputzarr(args.inputzarr)
filepathcheck.outputzarr(args.outputzarr)
args.intermediatezarr = filepathcheck.intermediatezarr(args.intermediatezarr)
args.config = filepathcheck.config(args.config)


###############################################

''' for testing
class ob: pass
args = ob()
args.latlonchunks = 4
args.inputzarr = ['/datadrive/hadgem3/nat_hist_zarr', '/datadrive/cam5/nat_hist_zarr']
args.intermediatezarr = ['/datadrive/hadgem3/trans5_intermediate', '/datadrive/cam5/trans5_intermediate']
args.outputzarr = ['/datadrive/hadgem3/trans5', '/datadrive/cam5/trans5']
args.max_mem = '20GB'
args.n_workers = 4
'''

# Set up cluster
if args.n_workers>1:
    client = Client(n_workers=args.n_workers, threads_per_worker=1,)# memory_limit=f'{args.max_mem//args.n_workers}GB')
    
datasets = [xr.open_zarr(f, consolidated=True) for f in args.inputzarr]

# Regrid if required
if args.config is not None:
    regridders = construct_regridders(*datasets, 
                                      args.config['resolution_match'], 
                                      args.config['scale_method'], 
                                      periodic=args.config['bbox'] is not None)

    # attributes are stripped by regridding module. Save them
    attrs = [{v:ds[v].attrs for v in ds.keys()} for ds in datasets]

    # regridders allow lazy evaluation
    datasets = [ds if rg is None else rg(ds).astype(np.float32) for ds, rg in zip(datasets, regridders)]
    del regridders
    
    # split at longitude
    datasets = [split_lon_at(ds, args.config['split_at']) for ds in datasets]
    
    # slice out area if required
    bbox = args.config['bbox']
    if bbox is not None:
        datasets = [ds.sel(lat=slice(bbox['S'], bbox['N']), lon=slice(bbox['W'], bbox['E'])) for ds in datasets]
    
    # reapply attributes
    for ds, atts in zip(datasets, attrs):
        for v, attr in atts.items():
            ds[v].attrs = attr


def parse_size(size):
    units = {"B": 1, "KB": 1e3, "MB": 1e6, "GB": 1e9}
    number, unit = re.findall(r'[A-Za-z]+|\d+', size)
    return int(float(number)*units[unit])

max_bytes = parse_size(args.max_mem)

for ds, outputzarr, temp_store in zip(datasets, args.outputzarr, args.intermediatezarr):
    for v in ds.variables:
        ds[v].encoding={}
    n_times = max(2*ds.nbytes//max_bytes, 1)
    dn = int(len(ds.time)/n_times + 1)
    
    mode="w-"
    append_dim=None
    for i in progressbar.progressbar(range(n_times)):
        tchunk = min(dn, len(ds.time) - i*dn)
        assert tchunk >=0, '`tchunk` < 0. somehting has gone wrong.'
        if tchunk==0:
            continue
        
        intermed_chunkdict = dict(lat=args.latlonchunks, 
                                  lon=args.latlonchunks, 
                                  time=tchunk, 
                                  run=len(ds.run))
        if 'height' in ds.keys():
            intermed_chunkdict['height']=1
        
        ds.isel(time=slice(i*dn, (i+1)*dn)) \
            .chunk(intermed_chunkdict) \
            .to_zarr(temp_store, mode=mode, append_dim=append_dim, consolidated=True)
        mode="a"
        append_dim = "time"
    
    load_chunks = {k:v[0] for k,v in ds.chunks.items()}
    load_chunks.update(dict(lat=args.latlonchunks, lon=args.latlonchunks, time=len(ds.time), run=len(ds.run)))
    ds = xr.open_zarr(temp_store, consolidated=True, chunks=load_chunks)
    for k in ds.keys(): del ds[k].encoding['chunks']
    with ProgressBar():
        ds.to_zarr(outputzarr, consolidated=True)
    
    os.system(f"rm -rf {temp_store}")

if args.n_workers>1:
    client.close()