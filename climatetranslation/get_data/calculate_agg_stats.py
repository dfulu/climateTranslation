import argparse
import xarray as xr
import dask
import numpy as np
from dask.diagnostics import ProgressBar


def get_dataset(zarr_path, filter_bounds=True):
    ds = xr.open_zarr(zarr_path, consolidated=True)
    if filter_bounds:
        ds = ds[[v for v in ds.data_vars if not 'bnds' in v]]
    return ds

def aggregates(ds, dim):
    """Function to calculate various aggregates to be used in pre-processing data."""
    agg = {}
    agg['mean'] = ds.mean(dim=dim)
    agg['std'] = ds.std(dim=dim)
    agg['min'] = ds.min(dim=dim)
    agg['max'] = ds.max(dim=dim)
    
    # need min for next section so do first round of computation now
    print('Calculating : mean, std, min, max')
    with ProgressBar(dt=10):
        (agg['mean'],agg['std'], 
         agg['min'], agg['max']) = dask.compute(*[agg[k] for k in ['mean','std','min','max']])
    
    log_ds = np.log(ds - agg['min'] + 1)
    agg['mean_log'] = log_ds.mean(dim=dim)
    agg['std_log'] = log_ds.std(dim=dim)
    agg['min_log'] = log_ds.min(dim=dim)
    agg['max_log'] = log_ds.max(dim=dim)
    
    # combine and compute
    print('Calculating : log-(mean, std, min, max)')
    with ProgressBar(dt=10):
        agg_ds = xr.concat([ds.assign_coords(aggregate_statistic=[k]) 
                                for k, ds in agg.items()], 
                           dim='aggregate_statistic').compute()
    agg_ds.attrs = ds.attrs
    agg_ds.attrs['log_calculation_note'] = "Calculated from log(x-x.min()+1)"
    return agg_ds

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--zarr_store', type=str, help='Path to the config file.')
    parser.add_argument('--output_path', type=str, help="outputs path. If ends in .nc saved to netcdf else zarr")
    args = parser.parse_args()
    
    ds = get_dataset(args.zarr_store, filter_bounds=True)
    agg_ds = aggregates(ds, ('time', 'run'))
    
    if args.output_path.endswith('.nc'):
        agg_ds.to_netcdf(args.output_path)
    else:
        agg_ds.to_zarr(args.output_path, consolidate=True)

