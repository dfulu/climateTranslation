import xarray as xr
import xesmf as xe
from climatetranslation.unit.data import _quick_add_bounds, _quick_remove_bounds
import argparse
import time

print('STARTING : ', time.ctime())
parser = argparse.ArgumentParser(description='Append zg netcdf files to zarr store.')
parser.add_argument('--files', type=str, nargs='+',
                       help='list of zg netcdf files')
parser.add_argument('--zarr_store', type=str, help='Filepath of zarr store to append to.')
args = parser.parse_args()

ds_main = xr.open_zarr(args.zarr_store)


chunk_dict = {'time': 1, 'run':1, 'height':1}

ds = xr.open_mfdataset(args.files)
ds = ds.sel(plev=50000)[['zg']] \
    .drop('plev') \
    .expand_dims({'run':[ds_main.run.item()]}) \
    .expand_dims({'height':[5500.]}) \
    .rename({'zg':'z500'}) \
    .transpose('run', 'time', 'height', 'lat', 'lon')
    

# regrid the data

ds_out = ds_main[['lat', 'lon']]
_quick_add_bounds(ds_out)
_quick_add_bounds(ds)
regridder = xe.Regridder(ds, ds_out, 'conservative', periodic=True)
regridder.clean_weight_file()
ds = regridder(ds)
_quick_remove_bounds(ds)

ds = ds_main.merge(ds).chunk(chunk_dict)
encoding={v: {'dtype': 'float32'} for v in ds.data_vars}

ds.to_zarr(args.zarr_store+'_z500_appended', consolidated=True,  encoding=encoding)