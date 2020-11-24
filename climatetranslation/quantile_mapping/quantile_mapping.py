import numpy as np
import xarray as xr

from scipy.interpolate import interp1d


def translate_quantile_value_single_month(ds, quantile_values, value2quantile=True):

    def interpolate(x, xs, ys):
        if x > xs[-1]:
            return 1 if value2quantile else ys[-1]
        elif x < xs[0]:
            return 0 if value2quantile else ys[0]
        return interp1d(xs, ys, kind="linear")(x)
    
    return xr.merge([
        xr.apply_ufunc(
            interpolate,
            ds[k],
            quantile_values[k] if value2quantile else quantile_values.quantiles,
            quantile_values.quantiles if value2quantile else quantile_values[k],
            input_core_dims=[[], ['quantiles'], ['quantiles']],
            exclude_dims = {'quantiles',},
            vectorize=True
        ).rename(k) for k in ds.keys()
    ])



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
        results = []
        for month, group in ds.groupby('time.month'):
            results.append(
                translate_quantile_value_single_month(
                    group, 
                    self.quantile_values.sel(month=month),
                    value2quantile=True
                )
            )
        return xr.concat(results, dim='time').drop('month').sortby('time')
    
    def inverse_transform(self, ds):
        results = []
        for month, group in ds.groupby('time.month'):
            results.append(
                translate_quantile_value_single_month(
                    group, 
                    self.quantile_values.sel(month=month),
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
        