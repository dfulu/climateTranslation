#!/usr/bin/env python  2415526
# coding: utf-8
import numpy as np
import xarray as xr

import cupy as cp
from cupyx.scipy.special import kl_div
from cupy.cuda import Device

import dask
from dask.diagnostics import ProgressBar
dask.config.set({'temporary_directory': '/home/s1205782/tmp'})

from climatetranslation.unit.utils import get_config
from climatetranslation.unit.data import (
    get_dataset, 
    construct_regridders,
    precip_kilograms_to_mm,
    dataset_time_overlap
)

import time
from tqdm import tqdm
import pickle
import os

xr.set_options(keep_attrs=True)

def add_title(ds, title):
    ds.attrs['title'] = title
    for k in ds.keys():
        ds[k].attrs['title'] = title

def clip(da, vmin, vmax):
    """A clip functrion which keeps attributes"""
    attrs = da.attrs
    x = da.clip(vmin, vmax)
    x.attrs = attrs
    return x

def transform(ds):
    ds = ds.isel(run=0)
    x = clip(ds.pr, None, 75)**.25
    y = ds.tas
    return x, y

def values_to_gpu(values):
    x = cp.array(values[0])
    y = cp.array(values[1])
    return x, y
        
def bin_data(test_values, ref_values, bins=100):
    test_values = values_to_gpu(test_values)
    ref_values = values_to_gpu(ref_values)
    
    def common_minmax(vals1, vals2):
        return  min(vals1.min(), vals2.min()), max(vals1.max(), vals2.max())
    
    minmaxes = [common_minmax(vals1, vals2) for vals1, vals2 in zip(ref_values, test_values)]
    bin_arrays = [cp.linspace(mn, mx, bins+1,endpoint=True) for mn, mx in minmaxes]
    
    def hist(values):
        x,y = values
        n = len(x)
        result = cp.histogram2d(x,y, bins=bin_arrays, density=False)[0].flatten()/n
        return result
    
    p = hist(test_values)
    q = hist(ref_values)
    return p, q
    
def est_kl_div(test_values, ref_values, bins=100, mask_p=False):
    p, q = bin_data(test_values, ref_values, bins=bins)
    if mask_p:
        mask = (q>0) & (p>0)
    else:
        mask = q>0
    q = q[mask]
    p = p[mask]
    div = kl_div(p, q).sum().get()
    return div

def est_js_div(test_values, ref_values, bins=100):
    p, q = bin_data(test_values, ref_values, bins=bins)    
    m = (p+q)/2
    div = (kl_div(p,m) + kl_div(q,m)).sum().get()/2
    return div

def est_energy_dist(test_values, ref_values, bins=100):
    p, q = bin_data(test_values, ref_values, bins=bins)
    dist = ((p - q)**2).sum().get()
    return dist


def this_slice(x_tup, lat, lon):
    return [x.sel(lat=lat, lon=lon) for x in x_tup]

if __name__=='__main__':
    conf = get_config("/home/s1205782/geos-fulton/model_outputs/outputs/v8.2_monsoon_hadgem_to_era5/config.yaml")
    test_set = True
    gpu_device = 0
    comp_kind = 'js_div' 
    
    metrics = dict(
        js_div=est_js_div,
        kl_div=est_kl_div,
        energy_dist=est_energy_dist,
    )
    est_div = metrics[comp_kind]
    
    conf_levels = conf['level_vars']
    bbox = conf['bbox']

    ds_target = get_dataset(f"/home/s1205782/geos-fulton/datadrive/era5/all_hist_global_zarr", 
                          conf_levels, 
                          filter_bounds=False,
                          split_at=360, 
                          bbox=bbox)

    ds_base = get_dataset(f"/home/s1205782/geos-fulton/datadrive/hadgem3/{'test_set/' if test_set else ''}all_hist_zarr", 
                          conf_levels, 
                          filter_bounds=False,
                          split_at=360, 
                          bbox=bbox)

    ds_trans = get_dataset(f"/home/s1205782/geos-fulton/datadrive/hadgem3/{'test_set/' if test_set else ''}v8.2_monsoon_to_era5_40k",
                           conf_levels, 
                           filter_bounds=False, 
                           split_at=360, 
                           bbox=bbox)

    ds_transqm = get_dataset(f"/home/s1205782/geos-fulton/datadrive/hadgem3/{'test_set/' if test_set else ''}v8.2_monsoon_unit40k_and_qm_to_era5",
                           conf_levels, 
                           filter_bounds=False, 
                           split_at=360, 
                           bbox=bbox)

    ds_qm = get_dataset(#f"/datadrive/hadgem3/all_hist_quantile_monsoon_zarr",
                        f"/home/s1205782/geos-fulton/datadrive/hadgem3/{'test_set/' if test_set else ''}all_hist_qm_to_era_monsoon_zarr",
                           conf_levels, 
                           filter_bounds=False, 
                           split_at=360, 
                           bbox=bbox)


    rg_t, rg_b = construct_regridders(
        ds_target, 
        ds_base, 
        resolution_match='downscale', 
        scale_method='conservative', 
        periodic=False)

    if rg_t is not None:
        ds_target = rg_t(ds_target)
    if rg_b is not None:
        ds_base = rg_b(ds_base)

    if conf['time_range'] is not None:
        if conf['time_range'] == 'overlap':
            ds_target, ds_base, ds_trans, ds_transqm, ds_qm = dataset_time_overlap([ds_target, ds_base, ds_trans, ds_transqm, ds_qm])
        elif isinstance(conf['time_range'], dict):
            time_slice = slice(conf['time_range']['start_date'], conf['time_range']['end_date'])
            ds_target, ds_base, ds_trans, ds_transqm, ds_qm = [ds.sel(time=time_slice) for ds in [ds_target, ds_base, ds_trans, ds_transqm, ds_qm]]
        else:
            raise ValueError("time_range not valid : {}".format(conf['time_range']))


    add_title(ds_target, "ERA5")
    add_title(ds_base, "HadGEM3")
    add_title(ds_trans, "UNIT")
    add_title(ds_transqm, "QM+UNIT")
    add_title(ds_qm, "QM")


    all_ds = [precip_kilograms_to_mm(ds) for ds in [ds_target, ds_base, ds_trans, ds_transqm, ds_qm]]

    with ProgressBar(dt=10) as bar:
        xs = dask.compute([transform(ds) for ds in all_ds])[0]

    print([x[0].time.shape for x in xs])

    # create array to store results
    titles = [x[0].attrs['title'] for x in xs]
    bins_list = [5, 10, 15, 20, 30, 40, 50, 100]
    results = xr.DataArray(
        np.zeros(
            ds_base.lat.shape
            +ds_base.lon.shape
            +(len(titles), len(bins_list))
        ), 
        coords=[
            ds_base.lat.values, 
            ds_base.lon.values, 
            titles,
            bins_list
        ],
        dims=['lat', 'lon', 'dataset', 'bins'],
        name=comp_kind,
    )


    titles = [x[0].attrs['title'] for x in xs]

    for i, lat in tqdm(enumerate(results.lat)):
        for j, lon in enumerate(results.lon):
            for k, x in enumerate(xs):
                test_values = this_slice(x, lat, lon)
                ref_values = this_slice(xs[0], lat, lon)
                dataset = x[0].attrs['title']
                for m, bins in enumerate(bins_list):
                    with Device(gpu_device):
                        div = est_div(test_values, ref_values, bins=bins)
                    results[i,j,k,m] = div
                    assert results.sel(lat=lat, lon=lon, dataset=dataset, bins=bins)==div

    results = results.to_dataset(name=comp_kind)
    results.to_netcdf(f"/home/s1205782/netcdf_store/{'test_set' if test_set else 'train_set'}_{comp_kind}_map.nc")
        