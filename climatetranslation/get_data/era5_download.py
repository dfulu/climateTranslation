"""
This python script is used to download daily data from ERA5.

References
----------
[1] https://climexp.knmi.nl/start.cgi
"""

import xarray as xr
import os
from climatetranslation.get_data.netcdfs_to_zarr import save_to_zarr
import argparse

parser = argparse.ArgumentParser(description='Download daily data from ERA5 and convert to zarr.')
parser.add_argument('--temp_store_dir', type=str, help='Directory used to temporarily store downloaded files.')
parser.add_argument('--zarr_store', type=str, help='Filepath of zarr store to create')
args = parser.parse_args()

os.makedirs(args.temp_store_dir, exist_ok=True)

# download all the files
for url in [
    "https://climexp.knmi.nl/ERA5/era5_t2m_daily_eu.nc",
    "https://climexp.knmi.nl/ERA5/era5_tp_daily_eu.nc",
    "https://climexp.knmi.nl/ERA5/era5_tmax_daily_eu.nc",
    "https://climexp.knmi.nl/ERA5/era5_tmin_daily_eu.nc",
    "https://climexp.knmi.nl/ERA5/era5_z500_daily_eu.nc"
]:
    os.system("wget -P {} {}".format(args.temp_store_dir, url))

# Some of these variables have 11:30 am timestamps and some have 12 noon timestamps
def times_to_noon(ds):
    newtime = ds.time.values.copy()
    mask = ((ds.time.dt.hour == 11) + (ds.time.dt.minute == 30))
    newtime[mask] = ds.time[mask].dt.ceil(freq='H').values
    ds['time'] = newtime
    return ds

def preprocess(ds):
    ds = times_to_noon(ds)
    if 'lev' in ds:
        ds = ds.isel(lev=0).drop('lev')
    return ds

ds = xr.open_mfdataset("{}/era5_*_daily_eu.nc".format(args.temp_store_dir), preprocess=preprocess)

# rename to match with other datasets
ds = ds.rename({'t2m':'tas', 'tmax':'tasmax', 'tmin':'tasmin', 'tp':'pr'})

# make sure lat is increasing
ds = ds.sortby(ds.lat)

# covert precip units from mm/day to kg m-2 s-1
ds['pr'] = ds.pr / (24*60**2)
ds.pr.attrs['units'] = 'kg m-2 s-1'

# Add to dimensions to match other data
ds = ds.expand_dims({'run':[1]})
ds = ds.transpose('run', 'time', 'lat', 'lon')

# chunk and save
save_to_zarr(ds, args.zarr_store)

os.system("rm {}/era5_*_daily_eu.nc".format(args.temp_store_dir))