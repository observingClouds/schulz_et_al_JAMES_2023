import os
import sys

import numpy as np
import pandas as pd
import tqdm
import xarray as xr
from intake import open_catalog
from omegaconf import OmegaConf

sys.path.append("./src/deps/")
from coat import organization as org  # noqa: E402

cfg = OmegaConf.load("./config/paths.cfg")
params = OmegaConf.load("./config/mesoscale_params.yaml")

cat = open_catalog(
    "https://raw.githubusercontent.com/eurec4a/eurec4a-intake/master/catalog.yml"
)

metrics_output_filename_fmt = cfg.ANALYSIS.MESOSCALE.METRICS.output_filename_fmt
sat_input_filename_fmt = cfg.OBS.SATELLITES["GOES16"].CH13.filename_fmt

geobounds = {}
geobounds["lat_min"] = params.metrics.geobounds.lat_min
geobounds["lat_max"] = params.metrics.geobounds.lat_max
geobounds["lon_min"] = params.metrics.geobounds.lon_min
geobounds["lon_max"] = params.metrics.geobounds.lon_max
org_settings = {}
org_settings[
    "threshold_discard_percentile"
] = params.metrics.BTbounds.threshold_discard_percentile
org_settings[
    "threshold_discard_temperature"
] = params.metrics.BTbounds.threshold_discard_temperature
org_settings[
    "threshold_cluster_llimit"
] = params.metrics.BTbounds.threshold_cluster_llimit
org_settings[
    "threshold_cluster_ulimit"
] = params.metrics.BTbounds.threshold_cluster_ulimit
org_settings["stencil"] = np.ones(
    (params.metrics.iorg.stencil[0], params.metrics.iorg.stencil[1])
)


def minimum_in_vicinity(data, window=None, min_periods=1):
    """Retrieve minimum data point in given vicinity."""
    assert isinstance(data, xr.DataArray)
    if window is None:
        window = {"lat": 4, "lon": 4}
    return data.rolling(window, min_periods=min_periods).min()


def cloudfraction(data, data_cloud_limits, **kwargs):
    """Calculate low-cloud cloud fraction.

    As high clouds potentially low level cloudiness, areas with high
    clouds in the vicinity are disregarded in the cloud fraction
    calculation.
    """
    min_vicinity = minimum_in_vicinity(data, **kwargs)
    sh_clouds = (
        (data < data_cloud_limits["threshold_cluster_ulimit"])
        & (data > data_cloud_limits["threshold_cluster_llimit"])
        & (min_vicinity > data_cloud_limits["threshold_cluster_llimit"])
    ).sum(dim=["lat", "lon"])
    clear_sky = ((data > data_cloud_limits["threshold_cluster_ulimit"])).sum(
        dim=["lat", "lon"]
    )
    CF = sh_clouds / (sh_clouds + clear_sky)
    return CF


## Simulations
experiment = 2
exp_longname = "LES_CampaignDomain_control"
for domain in [1, 2, 3]:
    sysnsat_input_filename_fmt = cat.simulations.ICON[exp_longname][
        f"rttov_DOM0{domain}"
    ]
    out_fn = metrics_output_filename_fmt.format(DOM=domain, type="rttov", exp=experiment)

    ds_rttov = sysnsat_input_filename_fmt.to_dask()
    data = ds_rttov[
        "synsat_rttov_forward_model_{DOM}__abi_ir__goes_16__channel_7".format(DOM=domain)
    ].sel(
        {
            "lat": slice(geobounds["lat_min"], geobounds["lat_max"]),
            "lon": slice(geobounds["lon_min"], geobounds["lon_max"]),
        }
    )
    cloud_fraction = cloudfraction(data, org_settings)
    percentile = np.nanpercentile(
        data, org_settings["threshold_discard_percentile"], axis=[1, 2]
    )

    ds = xr.Dataset(
        {
            "percentile_BT": ("time", percentile),
            "cloud_fraction": ("time", cloud_fraction.values),
        },
        coords={"time": data.time.values},
    )

    ds.to_netcdf(out_fn)

## Observations
for domain in [1, 2]:
    dates = pd.date_range(
        params.metrics.dates.start,
        params.metrics.dates.stop,
        freq=params.metrics.dates.step,
    )

    results_satellite = {}
    i = 0
    for date in tqdm.tqdm(dates):
        f = date.strftime(sat_input_filename_fmt.format(dom=domain))
        if not os.path.exists(f):
            continue
        ds = xr.open_dataset(f)
        data = ds.C13.squeeze()
        data = data.sel(
            {
                "lat": slice(geobounds["lat_max"], geobounds["lat_min"]),
                "lon": slice(geobounds["lon_min"], geobounds["lon_max"]),
            }
        )
        cloud_fraction = cloudfraction(data, org_settings).item(0)
        percentile = np.nanpercentile(data, org_settings["threshold_discard_percentile"])

        results_satellite[i] = {
            "domain": domain,
            "filename": f,
            "time": date,
            "cloud_fraction": cloud_fraction,
            "percentile_BT": percentile,
        }
        i += 1
    df = pd.DataFrame.from_dict(results_satellite, orient="index")
    df.to_xarray().to_netcdf(
        metrics_output_filename_fmt.format(DOM=domain, type="goes16", exp="")
    )
