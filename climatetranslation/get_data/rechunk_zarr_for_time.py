import xarray as xr
import zarr
from rechunker import rechunk

import argparse
import time
import os

from dask.distributed import Client
from dask.diagnostics import ProgressBar

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
args.n_times = 8
args.inputzarr = ['/datadrive/hadgem3/nat_hist_zarr', '/datadrive/cam5/nat_hist_zarr']
args.rechunkintermediatezarr = ['/datadrive/hadgem3/trans5_rechunk_intermediate', '/datadrive/cam5/trans5_rechunk_intermediate']
args.regridintermediatezarr = ['/datadrive/hadgem3/trans5_regrid_intermediate', '/datadrive/cam5/trans5_regrid_intermediate']
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

    
for ds, outputzarr, regrid_temp_store, rechunk_temp_store in zip(datasets, args.outputzarr, args.regridintermediatezarr, args.rechunkintermediatezarr):
    break
    # save out the regridded data if regridding is peformed
    if len(datasets)>1:
        with ProgressBar():
            ds.to_zarr(regrid_temp_store, consolidated=True)
        ds = xr.open_zarr(regrid_temp_store, consolidated=True)

    target_chunks = dict(run=1, time=1, lat=args.latlonchunks, lon=args.latlonchunks, height=1)
    target_chunks_dict = {k:tuple([target_chunks[d] for d in ds[k].dims]) for k in ds.keys()}

    # Compute rechunking method
    array_plan = rechunk(ds, target_chunks_dict, args.max_mem, outputzarr, temp_store=rechunk_temp_store)

    with ProgressBar():
        array_plan.execute()

    os.system(f"rm -rf {rechunk_temp_store}")
    if len(datasets)>1:
        os.system(f"rm -rf {regrid_temp_store}")

    # This next section manually consolidates the zarr store
    zarr.convenience.consolidate_metadata(outputzarr)

if args.n_workers>1:
    client.close()