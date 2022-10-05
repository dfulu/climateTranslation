echo "Converting to zarr"
python ~/repos/climateTranslation/get_data/netcdf_to_zarr.py --files $(ls netcdfs/*.nc) --zarr_store nat_hist_zarr
