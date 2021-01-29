import argparse
import numpy as np
from climatetranslation.diagnostics.simple_aggregates import aggregates
from climatetranslation.unit.data import get_dataset, dataset_time_overlap, construct_regridders
from climatetranslation.unit.utils import get_config


parser = argparse.ArgumentParser()
parser.add_argument('conf', type=str, help='Config file.')
args = parser.parse_args()
conf = get_config(args.conf)

ds_a = get_dataset(conf['data_zarr_a'], conf['level_vars'], filter_bounds=False, split_at=conf['split_at'], bbox=conf['bbox'])
ds_b = get_dataset(conf['data_zarr_b'], conf['level_vars'], filter_bounds=False, split_at=conf['split_at'], bbox=conf['bbox'])

periodic = conf['bbox'] is None
# match resolution of the pair
rg_a, rg_b = construct_regridders(ds_a, ds_b, conf['resolution_match'], conf['scale_method'], periodic)


if conf['time_range'] is not None:
    if conf['time_range'] == 'overlap':
        ds_a, ds_b = dataset_time_overlap([ds_a, ds_b])
    elif isinstance(conf['time_range'], dict):
        time_slice = slice(conf['time_range']['start_date'], conf['time_range']['end_date'])
        ds_a = ds_a.sel(time=time_slice)
        ds_b = ds_b.sel(time=time_slice)
    else:
        raise ValueError("time_range not valid : {}".format(conf['time_range']))

if rg_a is not None:
    ds_a = rg_a(ds_a).astype(np.float32)
if rg_b is not None:
    ds_b = rg_b(ds_b).astype(np.float32)

print('-'*8 + "CALCULATING FOR A" + '-'*8)
agg_ds_a = aggregates(ds_a, ('time', 'run'))
if conf['agg_data_a'].endswith('.nc'):
    agg_ds_a.to_netcdf(conf['agg_data_a'])
else:
    agg_ds_a.to_zarr(conf['agg_data_a'], consolidate=True)
    
print('-'*8 + "CALCULATING FOR B" + '-'*8)
agg_ds_b = aggregates(ds_b, ('time', 'run'))
if conf['agg_data_b'].endswith('.nc'):
    agg_ds_b.to_netcdf(conf['agg_data_b'])
else:
    agg_ds_b.to_zarr(conf['agg_data_b'], consolidate=True)