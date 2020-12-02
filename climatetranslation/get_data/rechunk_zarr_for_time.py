import xarray as xr
import zarr
from rechunker import rechunk

import argparse
import time
import os
import re

from dask.distributed import Client
from dask.diagnostics import ProgressBar

import progressbar

from climatetranslation.unit.data import construct_regridders

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
    
    def rechunkintermediatezarr(self, filepaths):
        if filepaths != '':
            if len(filepaths) != self.n_inputs:
                raise ValueError(f"intermediatezarr : {filepaths} must have same number of files as input zarr list  ({self.n_inputs}).")
        else:
            filepaths = [f + '_rechunk_intermediate' for f in self.outputzarrs]
        return filepaths
    
    def regridintermediatezarr(self, filepaths):
        if filepaths != '':
            if len(filepaths) != self.n_inputs:
                raise ValueError(f"regridintermediatezarr : {filepaths} must have same number of files as input zarr list  ({self.n_inputs}).")
        else:
            filepaths = [f + '_regrid_intermediate' for f in self.outputzarrs]
        return filepaths

    
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
parser.add_argument('--rechunkintermediatezarr', 
                    type=str, 
                    help=('Rechunking intermediate store filename(s) - either 1 or 2 zarr files, must be same number as input.'
                        + 'Defaults to [inputzarr]_rechunk_intermediate'), 
                    default=''
)
parser.add_argument('--regridintermediatezarr', 
                    type=str, 
                    help=('Regridding intermediate store filename(s) - either 1 or 2 zarr files, must be same number as input.'
                        + 'Defaults to [inputzarr]_regrid_intermediate'), 
                    default=''
)
parser.add_argument('--latlonchunks', type=int, help='New chunk size for both lat and lon', default=4)
parser.add_argument('--max_mem', type=lambda x: f"{x}GB", help='Max RAM usgae in GB', default='40')
parser.add_argument('--n_workers', type=int, help='Number of workers', default=1)
args = parser.parse_args()

filepathcheck = Filepathcheck()
filepathcheck.inputzarr(args.inputzarr)
filepathcheck.outputzarr(args.outputzarr)
args.rechunkintermediatezarr = filepathcheck.rechunkintermediatezarr(args.rechunkintermediatezarr)
args.regridintermediatezarr = filepathcheck.regridintermediatezarr(args.regridintermediatezarr)

###############################################

''' for testing
class ob: pass
args = ob()
args.latlonchunks = 4
args.inputzarr = ['/datadrive/hadgem3/nat_hist_zarr', '/datadrive/cam5/nat_hist_zarr']
args.intermediatezarr = ['/datadrive/hadgem3/trans5_intermediate', '/datadrive/cam5/trans5_intermediate']
args.outputzarr = ['/datadrive/hadgem3/trans5', '/datadrive/cam5/trans5']
args.max_mem = '20GB'
args.n_workers = 1
'''

# Set up cluster
if args.n_workers>1:
    client = Client(n_workers=args.n_workers, threads_per_worker=1,)# memory_limit=f'{args.max_mem//args.n_workers}GB')
    
datasets = [xr.open_zarr(f, consolidated=True) for f in args.inputzarr]

# regrid to common grid if nore than 1 dataset
if len(datasets)>1:
    regridders = [*construct_regridders(*datasets)]
    datasets = [ds if rg is None else rg(ds) for ds, rg in zip(datasets, regridders)]
    

def parse_size(size):
    units = {"B": 1, "KB": 1e3, "MB": 1e6, "GB": 1e9}
    number, unit = re.findall(r'[A-Za-z]+|\d+', size)
    return int(float(number)*units[unit])

max_bytes = parse_size(args.max_mem)   

for ds, outputzarr, temp_store in zip(datasets, args.outputzarr, args.intermediatezarr):
    n_times = 2*ds.nbytes//max_bytes
    dn = len(ds.time)//(n_times-1)
    
    mode="w-"
    append_dim=None
    for i in progressbar.progressbar(range(n_times)):
        ds.isel(time=slice(i*dn, (i+1)*dn)) \
            .chunk(dict(lat=args.latlonchunks, lon=args.latlonchunks, time=dn)) \
            .to_zarr(temp_store, mode=mode, append_dim=append_dim, consolidated=True)
        mode="a"
        append_dim = "time"
    
    load_chunks = {k:v[0] for k,v in ds.chunks.items()}
    load_chunks.update(dict(lat=args.latlonchunks, lon=args.latlonchunks,time=-1))
    ds = xr.open_zarr(temp_store, consolidated=True, chunks=load_chunks)

    with ProgressBar():
        ds.to_zarr(outputzarr, consolidated=True)

    os.system(f"rm -rf {temp_store}")

if args.n_workers>1:
    client.close()