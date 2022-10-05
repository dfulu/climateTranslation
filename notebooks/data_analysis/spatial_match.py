#!/usr/bin/env python
# coding: utf-8

import numpy as np
import xarray as xr

import cupy as cp
import cupyx.scipy.signal as signal

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

from datetime import datetime
from tqdm import tqdm
import os

from collections import UserDict
from functools import reduce

xr.set_options(keep_attrs=True)

################################################################################
# CONVENIENCE FUNCTIONS


def add_title(ds, title):
    ds.attrs['title'] = title
    for k in ds.keys():
        ds[k].attrs['title'] = title


def pr_transform(x):
    x = x.clip(None, 75)**0.25
    return x


################################################################################
# GPU MANAGEMENT
        
def cuda_release_memory():
    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()
    mempool.free_all_blocks()

def human_readable_bytes(bytes_int):
    for i, letter in enumerate('TGMK'):
        factor = 10**((4-i)*3)
        if bytes_int/factor>1:
            return f"{bytes_int/factor:.1f} {letter}B"
    return f"{bytes_int}  B"
    

def cuda_memory_use():
    mempool = cp.get_default_memory_pool()
    return f"Used: {human_readable_bytes(mempool.used_bytes())} | Total: {human_readable_bytes(mempool.total_bytes())}"

################################################################################
# COMMON MATCHING FUNCTIONS

def minargmin(x, axis=None):
    argmins = x.argmin(axis=axis)
    selection = [cp.arange(s) for s in x.shape]
    selection[axis] = argmins
    mins = x[tuple(selection)]
    return mins, argmins

def maxargmax(x, axis=None):
    argmaxs = x.argmax(axis=axis)
    selection = [cp.arange(s) for s in x.shape]
    selection[axis] = argmaxs
    maxs = x[tuple(selection)]
    return maxs, argmaxs

################################################################################
# MATCH METRICS

def mse_best_match(x_ref, x_search):
    scores = 0
    x = None
    for k in x_ref.keys():
        del x
        x = cp.array(x_ref[k], dtype=np.float32)[None, ...]
        y = x_search[k][:, None, ...]
        scores += ((x - y)**2).mean(axis=(-1,-2))
    s = minargmin(scores, axis=0)
    return s

def mae_best_match(x_ref, x_search):
    scores = 0
    x = None
    for k in x_ref.keys():
        del x
        x = cp.array(x_ref[k], dtype=np.float32)[None, ...]
        y = x_search[k][:, None, ...]
        scores += cp.abs(x - y).mean(axis=(-1,-2))
    s = minargmin(scores, axis=0)
    return s


def gaussian(window_size, sigma):
    gauss = cp.exp(-(cp.arange(window_size) - window_size//2)**2/float(2*sigma**2))
    return gauss/gauss.sum()

def create_window(window_size):
    _1D_window = gaussian(window_size, 1.5)[:, None]
    _2D_window = _1D_window.dot(_1D_window.T)
    return _2D_window[None, None]


def _ssim(img_a, img_b, window_size=11, R=1):
    
    K1 = 0.01**2
    K2 = 0.03**2
    
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2
    
    img1 = img_a[:,None]
    img2 = img_b[None]
    window = create_window(window_size)[None,...]
    
    mu1 = signal.fftconvolve(img1, window, mode='same', axes=(-1,-2))
    mu2 = signal.fftconvolve(img2, window, mode='same', axes=(-1,-2))

    mu1_mu2 = mu1*mu2
    sigma12 = signal.fftconvolve(img1*img2, window, mode='same', axes=(-1,-2)) - mu1_mu2
    
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))
    del sigma12, mu1_mu2
    cuda_release_memory()
    
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    
    sigma1_sq = signal.fftconvolve(img1*img1, window, mode='same', axes=(-1,-2)) - mu1_sq
    sigma2_sq = signal.fftconvolve(img2*img2, window, mode='same', axes=(-1,-2)) - mu2_sq
    
    sigs = sigma1_sq + sigma2_sq
    del sigma1_sq, sigma2_sq
    cuda_release_memory()
    
    mus = mu1_sq + mu2_sq
    del mu1_sq, mu2_sq
    cuda_release_memory()
    
    denom = (mus + C1)*(sigs + C2)
    del mus, sigs
    cuda_release_memory()
    
    ssim_map_norm = ssim_map/denom
    
    ssim_score = ssim_map_norm.mean(axis=(-1, -2, -3))
    
    del ssim_map, ssim_map_norm

    return ssim_score

def ssim_best_match(x_ref, x_search):
    scores = 0
    x = None
    for k in x_ref.keys():
        del x
        x = cp.array(x_ref[k], dtype=np.float32)[:, None]
        y = x_search[k][:, None]
        scores += _ssim(x, y, window_size=11)
    scores = scores/len(x_ref)
    return maxargmax(scores, axis=1)

################################################################################
# WRAPPER FOR APPLICATION OF ALL MATCH METRICS
    
def _best_matching_internal(ds_ref, ds_list, variables, transforms, func, N, ref_title, device_num):
    
    results_dict = NestedDict()
    variable_string = '+'.join(variables)
    
    def norm(x):
        mn = x.min()
        mx = x.max()
        return (x - mn)/(mx - mn)
    
    if func==ssim_best_match:
        transforms = {k: lambda x: norm(f(x)) for k, f in transforms.items()}
    
    with cp.cuda.Device(device_num):
        #print(f"\n\nDev.{device_num} (start) - {cuda_memory_use()}")
        
        y = {v:cp.array(ds_ref[v].isel(run=0).values, dtype=np.float32) for v in variables}
        y = {k:transforms[k](v) for k,v in y.items()}
        
        #print(f"Dev.{device_num} (add y) - {cuda_memory_use()}")
        
        for ds in ds_list:
            
            ds_title = ds.attrs['title']
            key = [ref_title, ds_title, func.__name__, variable_string]
            results_dict[key+['values']] = []
            results_dict[key+['positions']] = []
            
            x = {v:transforms[v](ds[v].isel(run=0).values) for v in variables}
            prefix = f"{ds_title} - {variable_string} - {func.__name__}()"
            for i in tqdm(range(int(np.ceil(list(x.values())[0].shape[0]/N)))):
                                    
                x_part = {k:v[i*N:i*N+N] for k,v in x.items()}

                #print(f"Dev.{device_num} (add x_part) - {cuda_memory_use()}")

                vals, pos = func(x_part, y)
                vals_cpu = vals.get()
                pos_cpu = pos.get()
                del vals, pos 
                cuda_release_memory() # keep

                results_dict[key+['values']] += [vals_cpu]
                results_dict[key+['positions']] += [pos_cpu]
            results_dict[key+['values']] = np.concatenate(results_dict[key+['values']])
            results_dict[key+['positions']] = np.concatenate(results_dict[key+['positions']])
        del x, y, x_part
        cuda_release_memory() # keep
        #print(f"Dev.{device_num} (exit loop) - {cuda_memory_use()}")
    cuda_release_memory()
    return results_dict

def best_matching(ds_list, ds_ref_list, variable_sets, variable_transforms, device_num):
    functions = [ssim_best_match, mse_best_match, mae_best_match]
    Ns = [1, 10, 10]

    results_dict_list = []
    
    for (N, func) in zip(Ns, functions):
        for ds_ref in ds_ref_list:
            ref_title = ds_ref.attrs['title']
            for variables in variable_sets:
                results_dict_list+=[_best_matching_internal(
                    ds_ref, ds_list, variables, variable_transforms, func, N, ref_title, device_num
                )]
    return results_dict_list



################################################################################
# DATA STRUCTURES TO STORE AND MANIPULATE RESULTS

class NestedDict(UserDict):
    def __setitem__(self, key, value):
        if isinstance(key, (list, tuple)):
            if key[0] not in self.keys():
                self[key[0]] = NestedDict()
            new_key = key[1] if len(key)==2 else key[1:]
            self[key[0]][new_key] = value 
        else:
            super().__setitem__(key, value)
        
    def __getitem__(self, key):
        if isinstance(key, (list, tuple)):
            new_key = key[1] if len(key)==2 else key[1:]
            return self[key[0]][new_key]             
        else:
            return super().__getitem__(key)
        
def merge_nested_dicts(a, b):
    assert type(a)==type(b)
    merged_dict = type(a)()
    for key in list(a)+list(b):
        if key in a and key in b:
            assert isinstance(a[key],  (dict, UserDict)), f"{a[key]} not a dict"
            assert isinstance(b[key],  (dict, UserDict)), f"{b[key]} not a dict"
            merged_dict[key] = merge_nested_dicts(a[key], b[key])
        elif key in a:
            merged_dict[key] = a[key]
        elif key in b:
            merged_dict[key] = b[key]
    return merged_dict


def nest_dict_to_nested_list(d):
    if not isinstance(d,  (dict, UserDict)):
        return d, None
    data_list = []
    keys_list = []
    for key, value in d.items():
        keys_list+=[key]
        datas, lower_keys = nest_dict_to_nested_list(value)
        data_list+=[datas]
    keys_list = [keys_list]
    if lower_keys is not None:
        keys_list = keys_list+lower_keys
    return data_list, keys_list

def nested_ragged_list_to_array(nested_list):
    if isinstance(nested_list, list):
        arrays_list = [nested_ragged_list_to_array(l) for l in nested_list]
        shapes = [x.shape for x in arrays_list]
        max_shape = tuple(max([s[i] for s in shapes]) for i in range(len(shapes[0])))
        new_arrays_list = []
        for array in arrays_list:
            new_array = np.full(max_shape, np.nan)
            new_array[tuple(slice(0, s) for s in array.shape)] = array
            new_arrays_list.append(new_array)
        return np.array(new_arrays_list)
    else:
        return np.array(nested_list)

def match_dict_to_dataset(d):
    data, labels = nest_dict_to_nested_list(d)
    data = nested_ragged_list_to_array(data)
    labels += [np.arange(data.shape[-1])]
    dimension_names = ['match_to', 'sample_from', 'match_func', 'variable', 'match', 'timestep']
    coords = {dim_name:locs for dim_name, locs in zip(dimension_names, labels)}
    return xr.Dataset(dict(data=(dimension_names, data)), coords=coords).data.to_dataset('match')

################################################################################


if __name__=='__main__':
    
    test_set = True # else train set
    gpu_device = 2
    append = '_v2'
    outroot = f"/home/s1205782/netcdf_store/{'test_set' if test_set else 'train_set'}"

    conf = get_config("/home/s1205782/geos-fulton/model_outputs/outputs/v8.2_monsoon_hadgem_to_era5/config.yaml")
    conf_levels = conf['level_vars']
    bbox = conf['bbox']
    hadgem_root = f"/home/s1205782/geos-fulton/datadrive/hadgem3{'/test_set' if test_set else ''}"
    
    variable_sets = [['pr'], ['z500'], ['tas'], ['pr', 'z500', 'tas']]
    variable_transforms = {'pr':pr_transform, 'z500':lambda x: x, 'tas':lambda x: x}

    print(datetime.now(), 'LOADING')

    ds_target = get_dataset(f"/home/s1205782/geos-fulton/datadrive/era5/all_hist_global_zarr", 
                          conf_levels, 
                          filter_bounds=False,
                          split_at=360, 
                          bbox=bbox)

    ds_base = get_dataset(f"{hadgem_root}/all_hist_zarr", 
                          conf_levels, 
                          filter_bounds=False,
                          split_at=360, 
                          bbox=bbox)


    ds_trans = get_dataset(f"{hadgem_root}/v8.2_monsoon_to_era5_40k",
                           conf_levels, 
                           filter_bounds=False, 
                           split_at=360, 
                           bbox=bbox)

    ds_transqm = get_dataset(f"/{hadgem_root}/v8.2_monsoon_unit40k_and_qm_to_era5",
                           conf_levels, 
                           filter_bounds=False, 
                           split_at=360, 
                           bbox=bbox)

    ds_qm = get_dataset(#f"/datadrive/hadgem3/all_hist_quantile_monsoon_zarr",
                        f"{hadgem_root}/all_hist_qm_to_era_monsoon_zarr",
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
    add_title(ds_transqm, "UNIT+QM")
    add_title(ds_qm, "QM")
    
    all_ds = [ds_target, ds_base, ds_trans, ds_transqm, ds_qm]
    all_ds = [precip_kilograms_to_mm(ds) for ds in all_ds]
    

    #with ProgressBar(dt=10):
    #    all_ds = dask.compute(all_ds)[0]
    
    comparison_datasets = all_ds[1:]
    target_datasets = all_ds[:1]
    
    print(datetime.now(), 'BEST MATCH TO OTHERS')
    # Calculate ERA's match to other datasets
    era_match_to_other_results = best_matching(target_datasets, comparison_datasets, variable_sets, variable_transforms, gpu_device)

    era_match_to_other_results = reduce(merge_nested_dicts, era_match_to_other_results)
    era_match_ds = match_dict_to_dataset(era_match_to_other_results)
    era_match_ds.to_netcdf(f"{outroot}_era_to_other_spatial_match{append}.nc")
    
    print(datetime.now(), 'BEST MATCH TO ERA')
    # Calculate other datasets' match to ERA
    dataset_match_to_era_results = best_matching(comparison_datasets, target_datasets, variable_sets, variable_transforms, gpu_device)

    dataset_match_to_era_results = reduce(merge_nested_dicts, dataset_match_to_era_results)
    other_match_ds = match_dict_to_dataset(dataset_match_to_era_results)
    other_match_ds.to_netcdf(f"{outroot}_other_to_era_spatial_match{append}.nc")

    
    
    del comparison_datasets, target_datasets, ds_target, ds_trans, ds_transqm, ds_qm
    
    print(datetime.now(), 'LOAD BASELINE')
    other_base_path = f"/home/s1205782/geos-fulton/datadrive/hadgem3/{'' if test_set else 'test_set/'}all_hist_zarr"
    ds_base_other = get_dataset(other_base_path, 
                          conf_levels, 
                          filter_bounds=False,
                          split_at=360, 
                          bbox=bbox)

    if rg_b is not None:
        ds_base_other = rg_t(ds_base_other)
        
    if conf['time_range'] is not None:
        if conf['time_range'] == 'overlap':
            ds_base, ds_base_other = dataset_time_overlap([ds_base, ds_base_other])
        elif isinstance(conf['time_range'], dict):
            time_slice = slice(conf['time_range']['start_date'], conf['time_range']['end_date'])
            ds_base_other = ds_base_other.sel(time=time_slice)
            
    add_title(ds_base_other, "HadGEM3_other")
    ds_base_other = precip_kilograms_to_mm(ds_base_other)
    
    with ProgressBar(dt=10):
        ds_base_other, ds_base = dask.compute([ds_base_other, ds_base])[0]
    
    print(datetime.now(), 'BEST MATCH BASELINE')
    # Calculate baseline for matching
    dataset_match_baseline_results = best_matching([ds_base], [ds_base_other], variable_sets, variable_transforms, gpu_device)

    dataset_match_baseline_results = reduce(merge_nested_dicts, dataset_match_baseline_results)
    baseline_match_ds = match_dict_to_dataset(dataset_match_baseline_results)
    baseline_match_ds.to_netcdf(f"{outroot}_baseline_spatial_match{append}.nc")
    