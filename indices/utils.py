
def crop_to_full_years(ds):
    """Crop a dataset to region of full years data."""
    
    # start and end time of sequence
    t0 = ds.time[0].item()
    tn = ds.time[-1].item()
    
    # calculate start and ends of years we have full coverage for
    if t0.month > 1 or t0.day > 1:
        t_start = t0.replace(year=t0.year+1, month=1, day=1)
    else:
        t_start = t0
    # Assume all calendars end on 30 Dec
    #  - imperfect as some calendars have 31 Dec but close enough
    if tn.month < 12 or tn.day < 30:
        t_end = tn.replace(year=tn.year-1, month=12, day=30)
    else:
        t_end = tn
        
    return ds.sel(time=slice(t_start, t_end))


def precip_to_mm(ds):
    """Convert precip to mm"""
    if ds.pr.sttrs['units']=='kg m-2 s-1':
        ds['pr'] = ds.pr * 24*60**2
        ds.pr.sttrs['units']='mm'
    elif ds.pr.sttrs['units']=='mm':
        pass
    else:
        raise ValueError('Unrecognised units')
    return ds