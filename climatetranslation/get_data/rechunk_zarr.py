import zarr
from rechunker import rechunk
from dask.diagnostics import ProgressBar


dz = zarr.open_consolidated('/datadrive/cam5/nat_hist_to_hadgem3_4ch_zarr')
target_chunks = dict(run=1, time=1, lat=192, lon=288)
target_chunks_dict = dict(tas=target_chunks, tasmin=target_chunks, tasmax=target_chunks, pr=target_chunks)
max_mem = "5GB"
target_store = "/datadrive/cam5/nat_hist_to_hadgem3_4ch_zarr_rechunked"
temp_store = "/datadrive/cam5/nat_hist_to_hadgem3_4ch_zarr_temp"

array_plan = rechunk(dz, target_chunks_dict, max_mem, target_store, temp_store)
with ProgressBar():
    array_plan.execute()