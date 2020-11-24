import numpy as np
import xarray as xr

import argparse
import progressbar
import os
import gc
import psutil
import time

from quantile_mapping import CDF
from climatetranslation.unit.utils import get_config
from climatetranslation.unit.data import (
    construct_regridders, 
    reduce_height, 
    get_dataset
)

print(f"staring - {time.asctime()}", flush=True)

# get config file
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Path to the config file.')
args = parser.parse_args()
config = get_config(args.config)

# unpack a few things for convenience
filepath = config['output_root']
eps = config['fraction_gap']
n_lat = config['number_splits_lat']
n_lon = config['number_splits_lon']
n_quantiles = config['number_of_quantiles']

# set up directory
os.makedirs(filepath, exist_ok=True)

# load the datasets
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


# calculate the quantiles
quantiles = np.linspace(eps, 1-eps, n_quantiles)

# chunk the files spatially
N_lat = len(ds_a.lat)
N_lon = len(ds_a.lon)
lat_chunks = np.linspace(0, N_lat, n_lat+1).astype(int)
lon_chunks = np.linspace(0, N_lat, n_lat+1).astype(int)

T0 = 0
def print_status(m):
    process = psutil.Process(os.getpid())
    global T0
    if T0 == 0:
        T0 = time.time()
    dt = time.time() - T0
    print(m+f" | memory: {process.memory_info().rss*1e-9:.2f}GB | time passed: {dt:.2f}s", flush=True) 

timerunslice = dict(run=slice(0, 8))

with progressbar.ProgressBar(max_value=n_lat*n_lon) as bar:
    for i in range(n_lat):
        for j in range(n_lon):
            # compute the CDFs on alt-lon chunks of data and save
            cdf_a = CDF(quantiles)
            cdf_b = CDF(quantiles)
            print_status(f'loading : a {i} {j}')
            ds_c = ds_a.copy(deep=True).isel(
                lat=slice(lat_chunks[i], lat_chunks[i+1]),
                lon=slice(lon_chunks[j], lon_chunks[j+1]), 
                **timerunslice
            ).compute()
            print_status(f'loaded - now fitting : a {i} {j}')
            cdf_a.fit(ds_c)
            cdf_a.save(f'{filepath}/a_{i}{j}.nc')
            ds_c.close()
            print_status(f'fitted : a {i} {j}')
            del ds_c, cdf_a
            gc.collect()
            print_status(f'loading : b {i} {j}')
            ds_c = ds_b.copy(deep=True).isel(
                lat=slice(lat_chunks[i], lat_chunks[i+1]),
                lon=slice(lon_chunks[j], lon_chunks[j+1]), 
                **timerunslice
            ).compute()
            print_status(f'loaded - now fitting : b {i} {j}')
            cdf_b.fit(ds_c)
            cdf_b.save(f'{filepath}/b_{i}{j}.nc')
            ds_c.close()
            print_status(f'fitted : b {i} {j}')
            del ds_c, cdf_b
            gc.collect()
            
            bar.update(i*n_lon+j+1)
            
# merge the files and clean up
cdf_a = CDF.load(f'{filepath}/a_*.nc')
cdf_b = CDF.load(f'{filepath}/b_*.nc')

cdf_a.save(f'{filepath}/quantiles_a.nc')
cdf_b.save(f'{filepath}/quantiles_b.nc')

os.system(f"rm {filepath}/a_*.nc")
os.system(f"rm {filepath}/b_*.nc")