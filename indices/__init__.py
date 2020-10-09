from . import etccdi
from . utils import crop_to_full_years, precip_to_mm
import progressbar
import xarray as xr


def apply_all_indices(ds, ds_ref):
    assert {'pr', 'tasmin', 'tas', 'tasmax'} - set(ds.keys())==set()
    assert{'tn10', 'tn90', 'tx10', 'tx90', 'pr95', 'pr99'} - set(ds_ref.keys())==set()
    ds = precip_to_mm(ds)
    ds = crop_to_full_years(ds)
    
    indices = []
    with progressbar.ProgressBar(max_value=27) as bar:
        bar.update(0)
        indices += [etccdi.fraction_of_frost_days(ds)]
        bar.update(1)
        indices += [etccdi.fraction_summer_days(ds)]
        bar.update(2)
        indices += [etccdi.fraction_of_icing_days(ds)]
        bar.update(3)
        indices += [etccdi.fraction_of_tropical_nights(ds)]
        bar.update(4)
        indices += [etccdi.growing_season_length(ds)]
        bar.update(5)
        indices += [etccdi.monthly_max_dayhigh_temp(ds)]
        bar.update(6)
        indices += [etccdi.monthly_max_daylow_temp(ds)]
        bar.update(7)
        indices += [etccdi.monthly_min_dayhigh_temp(ds)]
        bar.update(8)
        indices += [etccdi.monthly_min_daylow_temp(ds)]
        bar.update(9)
        indices += [etccdi.fraction_of_10perc_min_days(ds, ds_ref.tn10)]
        bar.update(10)
        indices += [etccdi.fraction_of_10perc_max_days(ds, ds_ref.tx10)]
        bar.update(11)
        indices += [etccdi.fraction_of_90perc_min_days(ds, ds_ref.tn90)]
        bar.update(12)
        indices += [etccdi.fraction_of_90perc_max_days(ds, ds_ref.tx90)]
        bar.update(13)
        indices += [etccdi.warm_speel_duration(ds, ds_ref.tx90)]
        bar.update(14)
        indices += [etccdi.cold_speel_duration(ds, ds_ref.tn10)]
        bar.update(15)
        indices += [etccdi.daily_temp_range(ds)]
        bar.update(16)
        indices += [etccdi.monthly_max_1day_precip(ds)]
        bar.update(17)
        indices += [etccdi.monthly_max_5day_precip(ds)]
        bar.update(18)
        indices += [etccdi.precip_intensity(ds)]
        bar.update(19)
        indices += [etccdi.precip_10mm_days(ds)]           
        bar.update(20)
        indices += [etccdi.precip_20mm_days(ds)]
        bar.update(21)
        # 22 is user defined criteria
        bar.update(22)
        indices += [etccdi.max_length_dry_spell(ds)]
        bar.update(23)
        indices += [etccdi.max_length_wet_spell(ds)]
        bar.update(24)
        indices += [etccdi.annual_heavy_precip(ds, ds_ref.pr95)]
        bar.update(25)
        indices += [etccdi.annual_very_heavy_precip(ds, ds_ref.pr99)]
        bar.update(26)
        indices += [etccdi.annual_precip(ds)]
        bar.update(27)
    return xr.merge(indices)
    
    
    
def prepare_reference(ds):
    assert {'pr', 'tasmin', 'tasmax'} - set(ds.keys())==set()
    ds = precip_to_mm(ds)
    ds = crop_to_full_years(ds)
    t10 = etccdi.doy_percentile(ds[['tasmin', 'tasmax']], q=10).rename({'tasmin':'tn10','tasmax':'tx10'}).drop('quantile')
    t90 = etccdi.doy_percentile(ds[['tasmin', 'tasmax']], q=90).rename({'tasmin':'tn90','tasmax':'tx90'}).drop('quantile')
    r95 = etccdi.doy_percentile(ds[['pr']], q=95).rename({'pr':'pr95'}).drop('quantile')
    r99 = etccdi.doy_percentile(ds[['pr']], q=99).rename({'pr':'pr99'}).drop('quantile')
    
    return xr.merge([t10, t90, r95, r99])
    
    