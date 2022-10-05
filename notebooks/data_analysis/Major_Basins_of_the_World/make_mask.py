import geopandas as gpd
from shapely.geometry import Point
import xarray as xr
import numpy as np
from tqdm import tqdm


gdf = gpd.read_file("Major_Basins_of_the_World.shp")

#ds = xr.open_dataset("example_file.nc")
ds = xr.open_zarr("/home/s1205782/geos-fulton/datadrive/era5/all_hist_global_zarr")

ds = ds[['lat', 'lon']]

mask = np.zeros((ds.lat.shape[0], ds.lon.shape[0], gdf.shape[0]), dtype=bool)
for i, lat in enumerate(tqdm(ds.lat.values)):
    for j, lon in enumerate(ds.lon.values):
        p  = Point(lon, lat)
        c = gdf.contains(p)
        mask[i,j, :] = c

ds['mask'] = xr.DataArray(
    mask, 
    dims=['lat', 'lon', 'basin'], 
    coords=[ds.lat, ds.lon, gdf.NAME.values], 
    name='mask'
)
ds = ds.groupby('basin').any(dim='basin')

ds.to_netcdf('/home/s1205782/netcdf_store/all_world_all_basin_masks_era5.nc')
print(mask.sum())
