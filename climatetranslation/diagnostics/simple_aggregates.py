import xarray as xr
import dask
import numpy as np
from dask.diagnostics import ProgressBar


def aggregates(ds, dim):
    """Function to calculate various aggregates to be used in pre-processing data."""
    
    # add 4th root of precip
    ds['pr_4root'] = ds['pr']**(1/4)
    
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
