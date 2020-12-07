import numpy as np
import xarray as xr

from scipy.interpolate import interp1d


def translate_quantile_value_single_month(ds, quantile_values, value2quantile=True):

    def interpolate(x, xs, ys):
        if value2quantile:
            fill_value=(0,1)
        else:
            fill_value=(ys[0], ys[-1])
        return interp1d(xs, ys, kind="linear", fill_value=fill_value, bounds_error=False, assume_sorted=True)(x)
    
    return xr.apply_ufunc(
            interpolate,
            ds,
            quantile_values if value2quantile else quantile_values.quantiles,
            quantile_values.quantiles if value2quantile else quantile_values,
            input_core_dims=[['time'], ['quantiles'], ['quantiles']],
            output_core_dims = [['time']],
            exclude_dims = {'quantiles',},
            dask= "allowed",
            vectorize=True
        )


class CDF:
    def __init__(self, quantiles):
        self.quantiles = quantiles
        
    def fit(self, ds):
        self.quantile_values = ds.groupby('time.month').quantile(
            self.quantiles, dim=('time', 'run'), interpolation='linear'
        ).rename(name_dict={'quantile':'quantiles'})
        
    def plot(self, lat, lon, figsize=(6, 18)):
        kn = len(self.quantile_values.keys())
        plt.figure(figsize=figsize)
        for i, k in enumerate(self.quantile_values.keys()):
            plt.subplot(kn, 1, i+1)
            self.quantile_values[k].sel(lat=lat, lon=lon, method='nearest').plot.line(x='quantiles',  add_legend=i==0)
        plt.tight_layout()

    def transform(self, ds):
        qs = self.quantile_values.sel(lat=ds.lat, lon=ds.lon)
        results = []
        for month, group in ds.groupby('time.month'):
            results.append(
                translate_quantile_value_single_month(
                    group, 
                    qs.sel(month=month),
                    value2quantile=True
                )
            )
        return xr.concat(results, dim='time').drop('month').sortby('time')
    
    def inverse_transform(self, ds):
        qs = self.quantile_values.sel(lat=ds.lat, lon=ds.lon)
        results = []
        for month, group in ds.groupby('time.month'):
            results.append(
                translate_quantile_value_single_month(
                    group, 
                    qs.sel(month=month),
                    value2quantile=False
                )
            )
        return xr.concat(results, dim='time').drop('month').sortby('time')
    
    def save(self, filepath):
        self.quantile_values.to_netcdf(filepath)
        
    @classmethod
    def load(cls, filepath):
        if '*' in filepath:
            quantile_values = xr.open_mfdataset(filepath).load()
        else:
            quantile_values = xr.load_dataset(filepath)
        quantiles = quantile_values.quantiles.values
        cdf = CDF(quantiles)
        cdf.quantile_values = quantile_values
        return cdf
            
    
class QauntileMapping:
    def __init__(self, cdfa, cdfb):
        self.cdfa = cdfa
        self.cdfb = cdfb
        
    def transform_a2b(self, ds):
        return self.cdfb.inverse_transform(self.cdfa.transform(ds))
        
    def transform_b2a(self, ds):
        return self.cdfa.inverse_transform(self.cdfb.transform(ds))
        