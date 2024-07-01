import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
from tqdm import tqdm
import cftime
import pandas as pd

## define some physical constants
RAD_PER_DEG = 2 * np.pi / 360  # radians per degree
R = 6.371e6  # radius of earth in m
M_PER_KM = 1000  # meters per km


def djf_avg(data):
    """function to trim data to Dec/Jan/Feb (DJF) months"""

    ## subset data
    data_seasonal_avg = data.resample(time="QS-DEC").mean()

    ## get annual average for DJF
    is_djf = data_seasonal_avg.time.dt.month == 12
    data_djf_avg = data_seasonal_avg.sel(time=is_djf)

    ## drop 1st and last averages, bc they only have
    ## 2 samples and 1 sample, respectively
    data_djf_avg = data_djf_avg.isel(time=slice(1, -1))

    ## Replace time index with year, corresponding to january.
    ## '+1' is because 'time's year uses december
    year = data_djf_avg.time.dt.year + 1
    data_djf_avg["time"] = year
    data_djf_avg = data_djf_avg.rename({"time": "year"})

    return data_djf_avg


def djf_avg_alt(data):
    """alternative (and slightly more general) function
    to trim data to Dec/Jan/Feb (DJF) months"""

    ## get 3-month rolling average average for DJF
    data_seasonal_avg = data_djf.resample({"time": "3MS"}, label="left").mean()

    ## subset for djf avg
    is_djf = data_seasonal_avg.time.dt.month == 12
    data_djf_avg = data_seasonal_avg.sel(time=is_djf)

    ## drop 1st and last averages, bc they only have
    ## 2 samples and 1 sample, respectively
    data_djf_avg = data_djf_avg.isel(time=slice(1, -1))

    ## Replace time index with year, corresponding to january.
    ## '+1' is because 'time's year uses december
    year = data_djf_avg.time.dt.year + 1
    data_djf_avg["time"] = year
    data_djf_avg = data_djf_avg.rename({"time": "year"})

    return data_djf_avg


def convert_longitude(longitude):
    """move longitude from range [0,360) to (-180,180].
    Function accepts and returns a numpy array representing longitude values"""

    ## find indices of longitudes which will become negative
    is_neg = longitude > 180

    ## change values at these indices
    longitude[is_neg] = longitude[is_neg] - 360

    return longitude


def switch_longitude_range(data):
    """move longitude for dataset or dataarray from [0,360)
    to (-180, 180]. Function accepts xr.dataset/xr.dataarray
    and returns object of same type"""

    ## get updated longitude coordinate
    updated_longitude = convert_longitude(data.lon.values)

    ## sort updated coordinate to be increasing
    updated_longitude = np.sort(updated_longitude)

    ## sort data ("reindex") according to update coordinate
    data = data.reindex({"lon": updated_longitude})

    return data


def reverse_latitude(data):
    """Change direction of latitude from [-90,90] to [90,-90]
    or vice versa"""

    lat_reversed = data.lat.values[::-1]
    data = data.reindex({"lat": lat_reversed})

    return data


def plot_setup(ax, lon_range, lat_range, xticks, yticks, scale):
    """Create map background for plotting spatial data.
    Returns modified 'ax' object."""

    ## specify transparency/linewidths
    grid_alpha = 0.0 * scale
    grid_linewidth = 0.5 * scale
    coastline_linewidth = 0.3 * scale
    label_size = 8 * scale

    ## crop map and plot coastlines
    ax.set_extent([*lon_range, *lat_range], crs=ccrs.PlateCarree())
    ax.coastlines(linewidth=coastline_linewidth)

    ## plot grid
    gl = ax.gridlines(
        draw_labels=True,
        linestyle="--",
        alpha=grid_alpha,
        linewidth=grid_linewidth,
        color="k",
        zorder=1.05,
    )

    ## add tick labels
    gl.bottom_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": label_size}
    gl.ylabel_style = {"size": label_size}
    gl.ylocator = mticker.FixedLocator(yticks)
    gl.xlocator = mticker.FixedLocator(xticks)

    return ax, gl


def make_cb_range(amp, delta):
    """Make colorbar_range for cmo.balance"""
    return np.concatenate(
        [np.arange(-amp, 0, delta), np.arange(delta, amp + delta, delta)]
    )


def spatial_avg(data, lon_range=[None, None], lat_range=[None, None]):
    """get global average of a quantity over the sphere.
    Data is xr.dataarray/xr.dataset with  a 'regular' lon/lat grid
    (equal lon/lat spacing between all data points)"""

    ## get latitude-dependent weights
    # first, convert latitude from degrees to radians
    # conversion factor = (2 pi radians)/(360 degrees)
    latitude_radians = data.lat * (2 * np.pi) / 360
    cos_lat = np.cos(latitude_radians)

    ## Next, trim data to specified range
    data_trim = data.sel(lon=slice(*lon_range), lat=slice(*lat_range))
    cos_lat_trim = cos_lat.sel(lat=slice(*lat_range))

    ## Next, compute weighted avg
    data_weighted = data_trim.weighted(weights=cos_lat_trim)
    data_avg = data_weighted.mean(["lat", "lon"])

    return data_avg


def spatial_int(data, lon_range=[None, None], lat_range=[None, None]):
    """compute spatial integral of a quantity on the sphere. For convenience,
    assume regular grid (constant lat/lon)"""

    ## Get latitude/longitude in radians.
    ## denote (lon,lat) in radians as (theta, phi)
    theta = data.lon * RAD_PER_DEG
    phi = data.lat * RAD_PER_DEG

    ## get differences for integration (assumes constant differences)
    dtheta = theta[1] - theta[0]
    dphi = phi[1] - phi[0]

    ## broadcast to grid
    dtheta = dtheta * xr.ones_like(theta)
    dphi = dphi * xr.ones_like(phi)

    ## Get area of patch
    dA = R**2 * np.cos(phi) * dphi * dtheta

    ## Integrate
    data_int = (data * dA).sum(["lat", "lon"])

    return data_int


def get_trend(data, dim="year"):
    """Get linear trend for an xr.dataarray along specified dimension"""

    ## Get coefficients for best fit
    polyfit_coefs = data.polyfit(dim=dim, deg=1)["polyfit_coefficients"]

    ## Get best fit line (linear trend in this case)
    trend = xr.polyval(data[dim], polyfit_coefs)

    return trend


def get_T_bar(t, trend, trend_type="exp"):
    """Get 'background' temperature"""

    ## get final and initial times
    tf = t[-1]
    ti = t[0]

    ## get "background" temperature
    if trend_type == "exp":
        a = 3e-2
        b = trend * (tf - ti) * np.exp(-a * (tf - ti))
        T_bar = b * np.exp(a * (t - ti))

    elif trend_type == "linear":
        T_bar = (t - t[0]) * trend

    else:
        print("Not a valid trend type")
        T_bar = 0.0

    return T_bar


def markov_simulation(
    ti, tf, dt=1 / 365.25, g=-0.3, n=0.3, n_members=1, trend=0, trend_type="exp"
):
    """Minimal version of the 'stochastic climate model' studied
    by Hasselman et al (1976) and Frankignoul and Hasselmann (1977).
    (See Eqn 3.6 in Frankignoul and Hasselman, 1976). The damping rate
    used here is much lower than in the paper, for illustration purposes.
    Args:
        - ti: number representing initial time (units: years)
        - tf: number representing final time (units: years)
        - dt: timestep (units: years)
        - g: damping rate (equivalent to lambda in the paper; units: 1/year)
        - n: noise amplitude (units: K / year^{1/2})
        - n_members: number of ensemble members
        - trend: calculated increase in "background" T, with units of 1/year
            (based on (T[tf]-T[ti]) / (tf-ti)
        - trend_type: one of "exp" (exponential) or "linear"
    """

    ## initialize RNG
    rng = np.random.default_rng()

    ## Get timesteps and dimensions for output
    t = np.arange(ti, tf, dt)
    nt = len(t)

    ## Create empty arrays arrays to hold simulation output
    T = np.zeros([n_members, nt])

    ## initialize with random value
    T[:, 0] = n * rng.normal(size=n_members)

    ## get "background" temperature
    T_bar = get_T_bar(t, trend, trend_type)[None, :]

    for i, t_ in tqdm(enumerate(t[:-1])):
        dW = np.sqrt(dt) * rng.normal(size=n_members)
        dT = g * (T[:, i] - T_bar[:, i]) * dt + n * dW
        T[:, i + 1] = T[:, i] + dT

    ##  put in xarray
    time_idx = xr.cftime_range(start=cftime.datetime(ti, 1, 1), periods=nt, freq="1D")
    e_member_idx = pd.Index(np.arange(1, n_members + 1), name="ensemble_member")

    T = xr.DataArray(
        T,
        dims=["ensemble_member", "time"],
        coords={"ensemble_member": e_member_idx, "time": time_idx},
    )

    ## resample to Annual
    T = T.resample({"time": "YS"}).mean()

    ## change time coordinate to year
    year = T.time.dt.year.values
    T = T.rename({"time": "year"})
    T["year"] = year

    return T
