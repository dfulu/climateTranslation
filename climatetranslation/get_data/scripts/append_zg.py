import xarray as xr
import xesmf as xe
from climatetranslation.unit.data import _quick_add_bounds, _quick_remove_bounds


chunk_dict = {'time': 1, 'run':1, 'height':1}

ds = xr.open_mfdataset("/datadrive/hadgem3/allhistzg/zg_*.nc")
ds= ds.sel(plev=50000)[['zg']] \
    .drop('plev') \
    .expand_dims({'run':[1]}) \
    .expand_dims({'height':[5500.]}) \
    .rename({'zg':'z500'}) \
    .transpose('run', 'time', 'height', 'lat', 'lon')
    

# regrid the data
ds_main = xr.open_zarr("/datadrive/hadgem3/all_hist_zarr").sel(run=[1,])
ds_out = ds_main[['lat', 'lon']]
_quick_add_bounds(ds_out)
_quick_add_bounds(ds)
regridder = xe.Regridder(ds, ds_out, 'conservative', periodic=True)
regridder.clean_weight_file()
ds = regridder(ds)
_quick_remove_bounds(ds)

ds = ds_main.merge(ds).chunk(chunk_dict)
encoding={v: {'dtype': 'float32'} for v in ds.data_vars}

ds.to_zarr("/datadrive/hadgem3/all_hist_zarr3", consolidated=True,  encoding=encoding)