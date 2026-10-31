"""Region definitions and data-loader stubs (ERA5-Land, FIRMS, SeasFire)."""
from __future__ import annotations

# (lon_min, lat_min, lon_max, lat_max), approximate bounding boxes [PROPOSAL; refine with shapefiles]
REGIONS = {
    "uttarakhand": (77.5, 28.7, 81.1, 31.5),
    "odisha": (81.3, 17.8, 87.5, 22.6),
    "california": (-124.5, 32.5, -114.1, 42.0),
}

ERA5L_VARS = ["2m_temperature", "2m_dewpoint_temperature", "volumetric_soil_water_layer_1",
              "10m_u_component_of_wind", "10m_v_component_of_wind", "total_precipitation"]


def bbox(region: str):
    return REGIONS[region.lower()]


def subset(ds, region: str, lon="longitude", lat="latitude"):
    """Subset an xarray Dataset to a region bbox (handles descending latitude)."""
    x0, y0, x1, y1 = bbox(region)
    lats = ds[lat].values
    ysl = slice(y1, y0) if lats[0] > lats[-1] else slice(y0, y1)
    return ds.sel({lon: slice(x0, x1), lat: ysl})


def load_era5_land(path, region=None, chunks="auto"):
    """Open local ERA5-Land NetCDF/Zarr (download via cdsapi separately)."""
    import xarray as xr
    ds = xr.open_zarr(path) if str(path).endswith(".zarr") else xr.open_mfdataset(path, chunks=chunks)
    return subset(ds, region) if region else ds


def load_firms(path, region=None):
    """Load a FIRMS archive CSV (MODIS/VIIRS) into a DataFrame, optionally clipped to region."""
    import pandas as pd
    df = pd.read_csv(path, parse_dates=["acq_date"])
    if region:
        x0, y0, x1, y1 = bbox(region)
        df = df[df.longitude.between(x0, x1) & df.latitude.between(y0, y1)]
    return df


def load_seasfire(path, region=None):
    """STUB: open the SeasFire Cube zarr (0.25 deg, 8-day). Variable selection TBD."""
    import xarray as xr
    ds = xr.open_zarr(path)
    return subset(ds, region) if region else ds


def vpd_kpa(t2m_k, d2m_k):
    """Vapour pressure deficit (kPa) from 2 m temperature and dewpoint (K), Tetens formula."""
    import numpy as np
    es = lambda t: 0.6108 * np.exp(17.27 * (t - 273.15) / (t - 273.15 + 237.3))
    return es(t2m_k) - es(d2m_k)
