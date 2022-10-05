"""ETCCDI indices calculation.

Derived from descriptions here:
http://etccdi.pacificclimate.org/list_27_indices.shtml?fbclid=IwAR0FN66bZwVBBmXVDyYNsf6nZ5W-rmTBo8jOH5Vt-ir0GyH0dDaeMLUtC1s
"""

import xarray as xr
import numpy as np

ZEROCELCIUS = 273.15


def fraction_of_frost_days(ds):
    """1. FD"""
    return (ds.tasmin < ZEROCELCIUS).mean(dim=('time', 'run')).rename('FD')

def fraction_summer_days(ds):
    """2. SU"""
    return (ds.tasmax > ZEROCELCIUS + 25).mean(dim=('time', 'run')).rename('SU')

def fraction_of_icing_days(ds):
    """3. ID"""
    return (ds.tasmax < ZEROCELCIUS).mean(dim=('time', 'run')).rename('ID')

def fraction_of_tropical_nights(ds):
    """4. TR"""
    return (ds.tasmin > ZEROCELCIUS + 20).mean(dim=('time', 'run')).rename('TR')

def _one_year_gsl(tas_year, apex_month=7, temp=5):
    
    def growing_days(x, **kwargs):
        x = np.nan_to_num(x, nan=0)
        return (x>temp+ZEROCELCIUS).sum(**kwargs)
    
    n_growing_days = tas_year.rolling(time=6, center=False, keep_attrs=True).reduce(growing_days)
    
    # first time we have 6 consecutive days of warm conditions
    warm_bool = n_growing_days==6
    first_gs = xr.apply_ufunc(np.nanargmax, warm_bool, input_core_dims=[['time']], kwargs={'axis': -1})
     # if no warm days set to max
    first_gs = first_gs.where(warm_bool.any(dim='time'), len(tas_year.time)-1)
    
    # first time we have 6 cold days after time_apex
    cold_bool = n_growing_days==0
    dt = tas_year.time[0].item().replace(month=apex_month, day=1)
    cold_bool = cold_bool.where(tas_year.time>=dt, other=False)
    last_gs = xr.apply_ufunc(np.nanargmax, cold_bool, input_core_dims=[['time']], kwargs={'axis': -1})
    last_gs = last_gs.where(cold_bool.any(dim='time'), other=len(tas_year.time)-1)
    
    # difference
    gs_length = last_gs - first_gs
    
    # if there is no growing season the length will be negative. Make zero
    gs_length = gs_length.where(gs_length>0, other=0)
    return gs_length

def growing_season_length(ds):
    """5. GSL"""
    ds_north = ds.tas.sel(lat=slice(0, 90))
    ds_south = ds.tas.sel(lat=slice(-90, 0-1e-8))
    
    # start and end time of sequence
    t0 = ds.time[0].item()
    tn = ds.time[-1].item()
    
    # calculate start and ends of years we have full coverage for
    if t0.month > 1 or t0.day > 1:
        dt_start_north = t0.replace(year=t0.year+1, month=1, day=1)
    else:
        dt_start_north = t0
    # Assume all calendars end on 30 Dec
    if tn.month < 12 or tn.day < 30:
        dt_end_north = tn.replace(year=tn.year-1, month=12, day=31)
    else:
        dt_end_north = tn
        
    gsl_north = ds_north.sel(time=slice(dt_start_north, dt_end_north)) \
                        .resample(time='1Y').map(_one_year_gsl, apex_month=7) \
                        .mean(dim=('time', 'run'))
    
    # calculate start and ends of years we have full coverage for
    if t0.month > 7 or (t0.month==7 and t0.day > 1):
        dt_start_south = t0.replace(year=t0.year+1, month=7, day=1)
    else:
        dt_start_south = t0.replace(year=t0.year, month=7, day=1)
    # Assume all calendars end on 30 Dec
    if tn.month < 6 or (tn.month == 6 and tn.day < 30):
        dt_end_south = tn.replace(year=tn.year-1, month=6, day=30)
    else:
        dt_end_south = tn.replace(year=tn.year, month=6, day=30)
    
    # offset southern hemisphere by 6 months
    offset = ds.time[0].item().replace(month=7, day=1) - ds.time[0].item().replace(month=1, day=1)
    gsl_south = ds_south.sel(time=slice(dt_start_south, dt_end_south)) \
                        .resample(time='1Y', loffset=offset) \
                        .map(_one_year_gsl, apex_month=1) \
                        .mean(dim=('time', 'run'))
    
    gsl = xr.concat([gsl_south, gsl_north], dim='lat').rename("GSL")
    
    return gsl

def monthly_max_dayhigh_temp(ds):
    """6. TX_x"""
    return (ds.tasmax.resample(time='1M').reduce(np.max)).groupby('time.month').mean(dim=('time', 'run')).rename('TX_x')

def monthly_max_daylow_temp(ds):
    """7. TN_x"""
    return (ds.tasmin.resample(time='1M').reduce(np.max)).groupby('time.month').mean(dim=('time', 'run')).rename('TN_x')

def monthly_min_dayhigh_temp(ds):
    """8. TX_n"""
    return (ds.tasmax.resample(time='1M').reduce(np.min)).groupby('time.month').mean(dim=('time', 'run')).rename('TX_n')

def monthly_min_daylow_temp(ds):
    """9. TN_n"""
    return (ds.tasmin.resample(time='1M').reduce(np.min)).groupby('time.month').mean(dim=('time', 'run')).rename('TN_n')

def doy_percentile(arr, q, window=5):
    def identity_(x, axis=None):
        return x
    return arr.rolling(time=window).reduce(identity_).groupby('time.dayofyear').quantile(q/100., skipna=True, dim=('time', '_rolling_dim', 'run'))

def fraction_of_10perc_min_days(ds, tn10):
    """10. TN10p"""
    return (ds.tasmin.groupby('time.dayofyear')<tn10).mean(dim=('time', 'run')).rename('TN10p')

def fraction_of_10perc_max_days(ds, tx10):
    """11. TX10p"""
    return (ds.tasmax.groupby('time.dayofyear')<tx10).mean(dim=('time', 'run')).rename('TX10p')

def fraction_of_90perc_min_days(ds, tn90):
    """12. TN90p"""
    return (ds.tasmin.groupby('time.dayofyear')>tn90).mean(dim=('time', 'run')).rename('TN90p')

def fraction_of_90perc_max_days(ds, tx90):
    """13. TX90p"""
    return (ds.tasmax.groupby('time.dayofyear')>tx90).mean(dim=('time', 'run')).rename('TX90p')

def warm_speel_duration(ds, tx90):
    """14. WSDI"""
    def inner(x, axis):
        return np.take(x, 0, axis=axis)*(6-5*np.take(x, 1, axis=axis))
    return (ds.tasmax.groupby('time.dayofyear')>tx90).rolling(time=6).reduce(np.all).rolling(time=2).reduce(inner).resample(time='1Y').reduce(np.sum).mean(dim=('time', 'run')).rename('WSDI')

def cold_speel_duration(ds, tn10):
    """15. CSDI"""
    def inner(x, axis):
        return np.take(x, 0, axis=axis)*(6-5*np.take(x, 1, axis=axis))
    return (ds.tasmin.groupby('time.dayofyear')<tn10).rolling(time=6).reduce(np.all).rolling(time=2).reduce(inner).resample(time='1Y').reduce(np.sum).mean(dim=('time', 'run')).rename('CSDI')

def daily_temp_range(ds):
    """16. DTR"""
    temp_diff = (ds.tasmax - ds.tasmin).resample(time='1M')
    return (temp_diff.reduce(np.mean)).groupby('time.month').mean(dim=('time', 'run')).rename('DTR')

def monthly_max_1day_precip(ds):
    """17. Rx1day"""
    return (ds.pr.resample(time='1M').reduce(np.max)).groupby('time.month').mean(dim=('time', 'run')).rename('Rx1day')

def monthly_max_5day_precip(ds):
    """18. Rx5day"""
    rolling_precip = ds.pr.rolling(time=5, center=False, keep_attrs=True).reduce(np.sum)
    max_roll_precip = rolling_precip.resample(time='1M').reduce(np.max)
    return max_roll_precip.groupby('time.month').mean(dim=('time', 'run')).rename("Rx5day")

def precip_intensity(ds):
    """19. SDII"""
    return ds.pr.where(ds.pr>1).mean(dim=('time', 'run')).rename('SDII')

def precip_10mm_days(ds):
    """20. R10mm"""
    return precip_nnmm_days(ds, 10)

def precip_20mm_days(ds):
    """21. R20mm"""
    return precip_nnmm_days(ds, 20)

def precip_nnmm_days(ds, nn):
    """22. Rnnmm"""
    return (ds.pr>=nn).mean(dim=('time', 'run')).rename(f'R{nn}mm')

def _roll_compare(x, i, axis):
    a = x[tuple([slice(None) if ax!=axis else slice(None, x.shape[axis]-i) for ax in range(len(x.shape))])]
    b = x[tuple([slice(None) if ax!=axis else slice(i, None) for ax in range(len(x.shape))])]
    return np.any((a-b)==0, axis=axis)

def _max_length_condition(ds_cond, axis=0):
    ds_cum_change = (~ds_cond).cumsum(axis=axis)
    max_sequence_length = np.zeros([ds_cond.shape[i] for i in range(len(ds_cond.shape)) if i!=axis])
    for i in range(1, ds_cond.shape[axis]):
        sequence = _roll_compare(ds_cum_change, i, axis)
        if sequence.sum()==0:
            break
        max_sequence_length[sequence]=i
    return max_sequence_length

def max_length_dry_spell(ds):
    """23. CDD"""
    return (ds.pr<1).resample(time='1Y').reduce(_max_length_condition).mean(dim=('time', 'run')).rename('CDD')

def max_length_wet_spell(ds):
    """24. CWD"""
    return (ds.pr>1).resample(time='1Y').reduce(_max_length_condition).mean(dim=('time', 'run')).rename('CWD')

def annual_heavy_precip(ds, pr95):
    """25. R95pTOT"""
    return ds.pr.where(ds.pr.groupby('time.dayofyear')>pr95).resample(time='1Y').sum().mean(dim=('time', 'run')).rename('R95pTOT')

def annual_very_heavy_precip(ds, pr99):
    """26. R99pTOT"""
    return ds.pr.where(ds.pr.groupby('time.dayofyear')>pr99).resample(time='1Y').sum().mean(dim=('time', 'run')).rename('R99pTOT')
    
def annual_precip(ds):
    """27. PRCPTOT"""
    return ds.pr.where(ds.pr>1).resample(time="1Y").sum().mean(dim=('time', 'run')).rename('PRCPTOT')