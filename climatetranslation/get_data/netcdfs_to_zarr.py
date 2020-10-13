"""
A command line tool to copy a list of files into a new zarr store.
"""

import os
import xarray as xr
import numpy as np
import re
import gc

################################################################################
# function library
################################################################################

def search_model_name(s):
    """Examine the filepath to infer the model name"""
    models_found = []
    if 'CAM5' in s:
        models_found.append('cam5')
    if 'HadGEM3' in s:
        models_found.append('hadgem3')
    if 'MIROC5' in s:
        models_found.append('miroc5')
    if len(models_found)!=1:
        raise valueError(f"Filename matches multiple models. Found {models_found}")
    return models_found[0]


def find_run_number(s, model):
    """Examine the filepath to infer the run number"""
    if model in ['cam5', 'miroc5']:
        run_num = int(re.search('_run([0-9]+)_', s).group(1))
    elif model=='hadgem3':
        run_num = int(re.search('_r1i1p([0-9]+)_', s).group(1))
    else:
        raise ValueError(f"Model '{model}' not found")
    return run_num


def split_by_run(files, model):
    """Group the files based on run number"""
    runs = [find_run_number(f, model) for f in files]
    unique_runs = np.unique(runs)
    grouped_files = [[f for r,f in zip(runs, files) if r==run] for run in unique_runs]
    return unique_runs, grouped_files


def open_datasets(run_numbers, grouped_files):
    """Open each model run and concatenate along a new 'run' axis"""
    ds_list = []
    print(f"    open_datasets()", flush=True)
    print(f"    opening:", flush=True)
    
    for i, (run_num, files) in enumerate(zip(run_numbers, grouped_files)):
        print(8*" "+f"run{run_num} : {len(files)} files", flush=True)
        
        da_list = []
        variables = np.unique([f.split('/')[1].split('_')[0] for f in files])
        
        for v in variables:
            print(12*" "+f"{v}", flush=True)
            
            da_list.append(xr.open_mfdataset([f for f in sorted(files) if f"/{v}_" in f], 
                                             coords='minimal', 
                                             compat='override',
                                             concat_dim="time",))
            
        print(12*" "+f"merging in", flush=True)
        ds_ = xr.merge(da_list).expand_dims({'run':[run_num]})
        if i>0:
            ds = xr.concat([ds, ds_], dim='run')
        else:
            ds = ds_
        
    ds = ds.sortby('run')
    return ds


def save_to_zarr(ds, filepath):
    """Save the dataset to local zarr store"""
    print(4*" "+'save_to_zarr()', flush=True)
    print(8*" "+'chunking', flush=True)
    chunk_dict = {'time': 1, 'run':1}
    ds = ds.chunk(chunk_dict)
    
    # Potential ToDo - Add compression
    # Probably only worth it chunked differently
    encoding={v: {'dtype': 'float32'} for v in ds.data_vars}
    
    print(8*" "+'saving', flush=True)
    ds.to_zarr(filepath, consolidated=True, encoding=encoding) 


def netcdf_files_to_zarr(files, filepath):
    """Take complete list of files and copy them into a new local zarr store"""
    print('setting up directory', flush=True)
    os.makedirs(filepath, exist_ok=False)
    
    # check all files are netcdf
    print('checking files format', flush=True)
    non_netcdf = [f for f in files if not f.endswith('.nc')]
    if len(non_netcdf)>0:
        raise ValueError(f"Found non-netcdf files: {non_netcdf}")

    # check for model name and make sure not multiple models
    print('checking if single model', flush=True)
    model = np.unique([search_model_name(f) for f in files])
    if len(model)!=1:
        raise ValueError(f"Input file list must contain exactly one model. Found {model}")
    else:
        model = model[0]
    
    # divide files into runs
    print('sorting files', flush=True)
    run_numbers, grouped_files = split_by_run(files, model)
    
    # open all the datasets
    print('opening datasets', flush=True)
    ds = open_datasets(run_numbers, grouped_files)
    
    print('filtering bounds', flush=True)
    ds = ds[[v for v in ds.data_vars if not 'bnds' in v]]
    
    print('saving to zarr', flush=True)
    save_to_zarr(ds, filepath)


################################################################################
# run
################################################################################

if __name__=='__main__':

    import argparse
    import time
    
    print('STARTING : ', time.ctime())
    parser = argparse.ArgumentParser(description='Convert netcdf files to zarr store.')
    parser.add_argument('--files', type=str, nargs='+',
                       help='list of netcdf files')
    parser.add_argument('--zarr_store', type=str, help='Filepath of zarr store to create')
    args = parser.parse_args()

    t0 = time.time()
    netcdf_files_to_zarr(args.files, args.zarr_store)
    print('FINISHED : ', time.time()-t0, 's')