import xarray as xr
import zarr
from rechunker import rechunk

import argparse
import time
import os

from dask.distributed import Client
from dask.diagnostics import ProgressBar

print(f"staring - {time.asctime()}", flush=True)

parser = argparse.ArgumentParser()
parser.add_argument('--latlonchunks', type=int, help='New chunk size fotr both lat and lon', default=4)
parser.add_argument('--inputzarr', type=str, help='Input filename', default='')
parser.add_argument('--intermediatezarr', type=str, help='Intermediate store filename. Defaults to [inputzarr]_intermediate', default='')
parser.add_argument('--outputzarr', type=str, help='Output filename', default='')
parser.add_argument('--max_mem', type=int, help='Max RAM usgae in GB', default=40)
parser.add_argument('--n_workers', type=int, help='Number of workers', default=1)
args = parser.parse_args()

''' for testing
class ob: pass
args = ob()
args.latlonchunks = 4
args.inputzarr = '/datadrive/hadgem3/nat_hist_zarr'
args.intermediatezarr = ''
args.outputzarr = '/datadrive/hadgem3/nat_hist_zarr_time_stream'
args.max_mem = 40
args.n_workers = 2
'''

# Construct parameters
if args.intermediatezarr=='':
    temp_store = args.outputzarr+'_intermediate'
else:
    temp_store = args.intermediatezarr

max_mem = f"{args.max_mem}GB"

# Set up cluster
if args.n_workers>1:
    client = Client(n_workers=args.n_workers, threads_per_worker=1,)# memory_limit=f'{args.max_mem//args.n_workers}GB')

# Load and define new chunks
ds = xr.open_zarr(args.inputzarr, consolidated=True)

target_chunks = dict(run=1, time=len(ds.time), lat=args.latlonchunks, lon=args.latlonchunks, height=1)
target_chunks_dict = {k:tuple([target_chunks[d] for d in ds[k].dims]) for k in ds.keys()}

# Compute rechunking method
array_plan = rechunk(ds, target_chunks_dict, max_mem, args.outputzarr, temp_store=temp_store)

with ProgressBar():
    array_plan.execute()
    
if args.n_workers>1:
    client.close()
    
os.system(f"rm -rf {temp_store}")

# This next section manually consolidates the zarr store
zarr.convenience.consolidate_metadata(args.outputzarr)

    
"""
#https://github.com/pangeo-data/rechunker/blob/master/rechunker/types.py
# https://github.com/pangeo-data/rechunker/blob/master/rechunker/executors/dask.py
# https://github.com/pangeo-data/rechunker/blob/master/rechunker/api.py

copy_specs, temp_group, target_group = rechunker.api._setup_rechunk(
    ds,
    target_chunks_dict,
    max_mem,
    args.outputzarr,
    temp_store=temp_store,
)
cs = copy_specs[5]

import dask
temp_group = zarr.group(temp_store)
target_group = zarr.group(args.outputzarr)
rechunker.api._setup_array_rechunk(
    dask.array.asarray(ds['tas']),
    target_chunks_dict['tas'],
    max_mem,
    target_group,
    temp_store_or_group=temp_group,
    name='tas',
)
"""