
climatetranslation
==================

Experiments on translating between different models of the climate


CC-BY-SA-4.0 License

--------------------

File Layout 
-----------


`Notebooks/` - Jupyter notebook files.
- `data_analysis/` - Notebooks used to explore model results.
- `development/` - Notebooks used to experiment and develop parts of the main library.

\
`climatetranslation/` - The main source library.
- `diagnostics/` - Modules used to explore the (translated) model data.
    - `indices/` - Functions for calculating standard extreme event indices.
    - `simple_aggregates.py` - Functions to calculate simple aggregate stats over large zarr files.
- `get_data/` - Scripts used to download model data from C20C+.
    - `wget_scipts/` - Lists of raw wget commands.
    - `create_data_download_script.py` - Command-line tool to download specific model data. This script filters the *wget\_scripts* to generate a new shell script. 
    - `netcdfs_to_zarr.py` - The downloaded data is in netcdf format. This command-line tool converts it to a zarr store.
- `unit/` - Adapted from the oiginal UNIT repo. Contains all modules required to train the climate2climate translation  network from datasets in zarr stores. See contained README for more details.

\
`scripts/` - Various command-line scripts built from main library functions.