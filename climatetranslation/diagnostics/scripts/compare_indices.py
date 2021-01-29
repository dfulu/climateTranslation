import argparse
from climatetranslation.diagnostics.indices import apply_all_indices, prepare_reference
from climatetranslation.unit.data import get_dataset


parser = argparse.ArgumentParser()
parser.add_argument('--zarr_store', type=str, help='Path to the config file.')
parser.add_argument('--output_path', type=str, help="Outputs path. If ends in .nc saved to netcdf else zarr")
args = parser.parse_args()

ds = get_dataset(args.zarr_store, filter_bounds=True)
agg_ds = aggregates(ds, ('time', 'run'))

if args.output_path.endswith('.nc'):
    agg_ds.to_netcdf(args.output_path)
else:
    agg_ds.to_zarr(args.output_path, consolidate=True)