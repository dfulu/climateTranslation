import numpy as np
import xarray as xr

import argparse
import progressbar

from quantile_mapping import CDF
from climatetranslation.unit.utils import get_config


quantiles = np.linspace()

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Path to the config file.')
parser.add_argument('--output_zarr', type=str, help="output zarr store path")
parser.add_argument('--x2x', type=check_x2x, help="any of [a2b, b2a, b2a, b2b]")
args = parser.parse_args()