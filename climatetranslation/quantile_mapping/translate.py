import numpy as np
import xarray as xr

import argparse
from dask.diagnostics import ProgressBar
import time

from climatetranslation.unit.utils import get_config
from climatetranslation.unit.data import (
    construct_regridders, 
    reduce_height, 
    get_dataset
)
from quantile_mapping import CDF, QauntileMapping

print(f"staring - {time.asctime()}", flush=True)


################################################
# parse arguments

def check_x2x(x2x):
    x2x = str(x2x)
    if x2x not in ['a2b', 'b2a']:
        raise ValueError("Invalid x2x arg")
    return x2x

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Path to the config file.')
parser.add_argument('--output_zarr', type=str, help="Output zarr store path")
parser.add_argument('--x2x', type=check_x2x, help="Any of [a2b, b2a]")
args = parser.parse_args()

config = get_config(args.config)

# unpack a few things for convenience
CDFfilepath = config['output_root']


################################################
# load and regrid data

ds_a = get_dataset(config['data_zarr_a'], config['level_vars'])
ds_b = get_dataset(config['data_zarr_b'], config['level_vars'])
rg_a, rg_b = construct_regridders(ds_a, ds_b)

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
    
ds = ds_a if args.x2x=='a2b' else ds_b


################################################
# load and construct quantile mapping model
    
cdf_a = CDF.load(f'{CDFfilepath}/quantiles_a.nc')
cdf_b = CDF.load(f'{CDFfilepath}/quantiles_b.nc')

QM = QauntileMapping(cdf_a, cdf_b)

transform_func = QM.transform_a2b if args.x2x=='a2b' else QM.transform_b2a

###############################################
# apply quantile mapping translation
        
result = ds.map_blocks(
    transform_func,
    template=ds,
)

print('setup complete - starting computation')

with ProgressBar():
        # append to zarr
        result.to_zarr(
            output_zarr, 
            consolidated=True
        )