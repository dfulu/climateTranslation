"""
This python script is used to download hourly data from the ERA5 archive.

References
----------
[1] https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land?tab=overview

https://cds.climate.copernicus.eu/cdsapp#!/dataset/derived-near-surface-meteorological-variables?tab=overview
"""

import cdsapi
import argparse
import xarray as xr
import os
import numpy as np

variables = {
    'tas': ['2m_temperature'],
    'pr': ['total_precipitation'],
    'all': ['2m_temperature', 'total_precipitation']
}

def check_variable(st):
    st = str(st)
    if st not in variables.keys():
        raise argparse.ArgumentTypeError("variable {} not in {}.".format(st, list(variables.keys())))
    return st

class check_range:
    def year_in_valid(self, year):
        y = int(year)
        if y<1981 or y>2020:
            raise argparse.ArgumentTypeError("Must be in range 1981-2020. User argument : {}".format(y))
    
    def check_start(self, year):
        self.year_in_valid(year)
        self.start_year = int(year)
        return int(year)
    
    def check_end(self, year):
        self.year_in_valid(year)
        if (int(year) - self.start_year)>=5:
            raise argparse.ArgumentTypeError("Year range {}-{} too large for download. Specify a maximum of 5 year inclusive range.".format(self.start_year, year))
        if (int(year) - self.start_year)<0:
            raise argparse.ArgumentTypeError("End year must be after start year.")
        return int(year)


year_range = check_range()
    
parser = argparse.ArgumentParser(description='Download ERA5 hourly data.')
parser.add_argument('--variables', 
                    type=check_variable, 
                    help="Variable from : {}[tas]".format(list(variables.keys())), default='tas')
parser.add_argument('--start_year', type=year_range.check_start, help='First year of download.')
parser.add_argument('--end_year', type=year_range.check_end, help='Last year of download.')
args = parser.parse_args()

    
request_dict = {
        'format': 'netcdf',
        'year': [str(y) for y in range(args.start_year, args.end_year+1)],
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
        'day': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
            '13', '14', '15',
            '16', '17', '18',
            '19', '20', '21',
            '22', '23', '24',
            '25', '26', '27',
            '28', '29', '30',
            '31',
        ],
        'area': [71.5, -25, 25, 45],
}

# reduce for testing
request_dict.update({
    'month': ['01'],
        'day': ['01', '02', '03'],
})
    
c = cdsapi.Client()
def f(*args, **kwargs):
    return
c.retrieve = f
interim_filenames = []

if args.variables in ['tas', 'all']:
    
    request_dict.update({
        'variable': variables['tas'],
        'time': [
            '00:00', '01:00', '02:00',
            '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00',
            '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00',
            '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00',
            '21:00', '22:00', '23:00',
        ],
    })
    
    filename = "{}-{}_{}.nc".format(args.start_year, args.end_year, 'tas_interim')
    interim_filenames.append(filename)
    
    c.retrieve(
        'reanalysis-era5-land',
        request_dict,
        filename
        )
    
    ds = xr.open_dataset(filename)
    print(ds.time.max(), ds.time.min())
    dg = ds.resample(dict(time="1D"), closed="left", label="left", keep_attrs=True)
    ds_min = dg.min().rename({k:k+'min' for k in ds.keys()})
    ds_max = dg.max().rename({k:k+'max' for k in ds.keys()})
    ds_mean = dg.mean().rename({k:k+'mean' for k in ds.keys()})
    dsm = xr.merge([ds_min, ds_max, ds_mean])
    
if args.variables in ['pr', 'all']:
    
    request_dict.update({
        'variable': variables['pr'],
        'time': [
            '00:00',
        ],
    })
    
    filename = "{}-{}_{}.nc".format(args.start_year, args.end_year, 'pr_interim')
    interim_filenames.append(filename)
    
    c.retrieve(
        'reanalysis-era5-land',
        request_dict,
        filename)
    
    ds = xr.open_dataset(filename)

    if ds.time.dtype == 'O':
        ds['time'] = ds.time - datetime.timedelta(days=1)
    elif ds.time.dtype == '<M8[ns]':
        ds['time'] = ds.time - np.timedelta64(1,'D')
        
    if args.variables =='all':
        dsm = xr.merge([ds, dsm])
    else:
        dsm = ds

dsm.to_netcdf("{}-{}_{}.nc".format(args.start_year, args.end_year, args.variables))

for f in interim_filenames:
    os.system("rm {}".format(f))