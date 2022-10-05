import numpy as np
import xarray as xr

import argparse
from dask.diagnostics import ProgressBar
import time
from dask.distributed import Client

from climatetranslation.unit.utils import get_config
from climatetranslation.unit.data import (
    construct_regridders, 
    reduce_height, 
    get_dataset,
    split_lon_at
)

from quantile_mapping import CDF

print(f"staring - {time.asctime()}", flush=True)
################################################
# parse arguments

def check_x(x):
    x = str(x)
    if x not in ['a', 'b']:
        raise ValueError("Invalid dataset_letter arg")
    return x

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Path to the config file.')
parser.add_argument('--output_zarr', type=str, help="Output zarr store path")
parser.add_argument('--dataset_letter', type=check_x, help="Any of [a, b]")
args = parser.parse_args()

config = get_config(args.config)

# unpack a few things for convenience
CDFfilepath = config['output_root']

################################################
# load and regrid data

ds_a = get_dataset(config['data_zarr_a'], config['level_vars'])
ds_b = get_dataset(config['data_zarr_b'], config['level_vars'])
rg_a, rg_b = construct_regridders(ds_a, ds_b, 
    resolution_match=config['resolution_match'],
    scale_method=config['scale_method'], 
    periodic=(config['bbox'] is None)
)

# attributes are stripped by regridding module. Save them
a_attrs = {v:ds_a[v].attrs for v in ds_a.keys()}
b_attrs = {v:ds_b[v].attrs for v in ds_b.keys()}

# regridders allow lazy evaluation
ds_a = ds_a if rg_a is None else rg_a(ds_a).astype(np.float32)
ds_b = ds_b if rg_b is None else rg_b(ds_b).astype(np.float32)

del rg_a, rg_b

# reapply attributes
for v, attr in a_attrs.items():
    ds_a[v].attrs = attr
for v, attr in b_attrs.items():
    ds_b[v].attrs = attr
    
ds = ds_a if args.dataset_letter=='a' else ds_b

# split at longitude
ds = split_lon_at(ds, config['split_at'])
    
# slice out area if required
print(config['bbox'])
if config['bbox'] is not None:
    bbox = config['bbox']
    ds = ds.sel(
        lat=slice(bbox['S'], bbox['N']), 
        lon=slice(bbox['W'], bbox['E'])
    )


################################################
# load cumulative distribution data
    
cdf = CDF.load(f'{CDFfilepath}/quantiles_{args.dataset_letter}.nc')

transform_func = cdf.transform

###############################################
# map to quantiles

if __name__=="__main__":
    result = ds.map_blocks(
        transform_func,
        template=ds,
    )
    
    client = Client(processes=False)

    print('setup complete - starting computation')

    with ProgressBar():
            # append to zarr
            result.to_zarr(
                args.output_zarr, 
                consolidated=True
            )