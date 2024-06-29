import xarray as xr
import numpy as np
import src.utils


def trim_to_north_atlantic(data):
    """convenience function to trim data north atlantic domain"""

    return data.sel(lon=slice(-70, 15), lat=slice(0, 70))


def trim_to_azores(data):
    """convenience function to trim data to Azores lon/lat range."""

    return data.sel(lon=slice(-60, 10), lat=slice(10, 52))
    # return data.sel(lon=slice(-65, 15), lat=slice(15, 65))


def compute_AHA(slp, slp_global_avg, norm_type="global_mean"):
    """compute Azores High Area index, similar to Cresswell-Clay et al. (2022).
    Defined as: area of Azores region which has (normalized) SLP exceeding 0.5 std
    of long term average. Normalized SLP is defined as local SLP minus
    globally-averaged SLP.

    Args:
    - slp is gridded SLP data (at global scale)
    - norm_type is one of {None, "global_mean", "detrend"} specifying
        how to normalize the AHA index

    Note: returns area in KM^2.
    """

    ## get SLP anomaly in Azores regions
    slp_azores = trim_to_azores(slp)

    ## get normalized anomaly (func of lon/lat)
    if norm_type == "global_mean":
        slp_azores_norm = slp_azores - slp_global_avg

    elif norm_type == "detrend":
        trend = src.utils.get_trend(slp_global_avg)
        slp_azores_norm = slp_azores - trend

    elif norm_type is None:
        slp_azores_norm = slp_azores

    else:
        print("Error: specify valid normalization type.")

    ## Get standard deviation (func of lon/lat)
    slp_azores_mean = slp_azores_norm.mean("year")
    slp_azores_std = slp_azores_norm.std("year")

    ## Get mask of grid cells exceeding 0.5 std threshold
    threshold = slp_azores_mean + 0.5 * slp_azores_std
    exceeds_thresh = slp_azores_norm > threshold

    ## Sum area of lon/lat cells exceeding threshold
    ## convert from True/False to 1.0/0.0 for integration
    AHA = src.utils.spatial_int(exceeds_thresh.astype(float))

    ## convert from m^2 to km^2
    m2_per_km2 = src.utils.M_PER_KM**2
    AHA *= 1 / m2_per_km2

    return AHA


def count_extremes(AHA, cutoff_perc=90.0, window=25):
    """Get rolling count of Azores High extreme events.
    Args:
    - cutoff_perc is percentile value in range (0 and 100) used to define
        'extreme' events
    - window is an integer specifying how many years the rolling window is.
    """

    ## get threshold for extreme events
    threshold = AHA.quantile(q=cutoff_perc / 100)

    ## Get boolean array: True if AHA exceeds thresh
    exceeds_thresh = AHA > threshold

    ## Get rolling count
    rolling_count = exceeds_thresh.rolling(dim={"year": window}, center=True).sum()

    ## remove NaN values at beginning and end
    nan_count = np.round((window - 1) / 2).astype(int)
    rolling_count = rolling_count.isel(year=slice(nan_count, -nan_count))

    return rolling_count


def count_extremes_wrapper(slp_data, norm_type="detrend", window=25):
    """wrapper function which takes in SLP dataset and computes # of Azores High extremes.
    Assumes dataset 'slp_data' contains pre-computed global mean"""

    ## extract global mean
    slp_trim = slp_data["slp"]
    slp_global_avg = slp_data["slp_global_avg"]

    ## Get AHA
    AHA = compute_AHA(slp_trim, slp_global_avg, norm_type=norm_type)

    ## count # of extremes
    return count_extremes(AHA, cutoff_perc=90.0, window=window)
