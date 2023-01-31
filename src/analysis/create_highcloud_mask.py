import glob
import os

import numpy as np
import pandas as pd
import xarray as xr
from intake import open_catalog
from omegaconf import OmegaConf

overwrite = False
fn = "../../data/intermediate/Quantile_brightnessT_GOES16.nc"

cfg = OmegaConf.load("../../config/paths.cfg")
params = OmegaConf.load("../../config/mesoscale_params.yaml")
cat_tmp = open_catalog(
    "https://raw.githubusercontent.com/eurec4a/eurec4a-intake/master/catalog.yml"
)

geobounds = {}
geobounds["lat_min"] = params.metrics.geobounds.lat_min
geobounds["lat_max"] = params.metrics.geobounds.lat_max
geobounds["lon_min"] = params.metrics.geobounds.lon_min
geobounds["lon_max"] = params.metrics.geobounds.lon_max

threshold_discard_percentile = params.metrics.BTbounds.threshold_discard_percentile
threshold_discard_temperature = params.metrics.BTbounds.threshold_discard_temperature

ds_bt_dom01 = cat_tmp.simulations.ICON.LES_CampaignDomain_control.rttov_DOM01.to_dask()
ds_bt_dom02 = cat_tmp.simulations.ICON.LES_CampaignDomain_control.rttov_DOM02.to_dask()
ds_goes16 = xr.open_mfdataset(
    sorted(
        glob.glob(
            cfg.OBS.SATELLITES.GOES16.CH13.filename_fmt_glob.format(date="*", time="*")
        )
    ),
    parallel=True,
)

print("Data read; starting selection")

BT_DOM01 = ds_bt_dom01.sel(
    lon=slice(geobounds["lon_min"], geobounds["lon_max"]),
    lat=slice(geobounds["lat_min"], geobounds["lat_max"]),
).synsat_rttov_forward_model_1__abi_ir__goes_16__channel_7
BT_DOM02 = ds_bt_dom02.sel(
    lon=slice(geobounds["lon_min"], geobounds["lon_max"]),
    lat=slice(geobounds["lat_min"], geobounds["lat_max"]),
).synsat_rttov_forward_model_2__abi_ir__goes_16__channel_7
BT_ABI = ds_goes16.sel(
    lon=slice(geobounds["lon_min"], geobounds["lon_max"]),
    lat=slice(geobounds["lat_max"], geobounds["lat_min"]),
).C13

print("Calculate ICON quantiles")

q_DOM01 = BT_DOM01.quantile(
    threshold_discard_percentile / 100, dim=["lat", "lon"]
).compute()
q_DOM02 = (
    BT_DOM02.chunk({"lat": 534, "lon": 533, "time": 12})
    .quantile(threshold_discard_percentile / 100, dim=["lat", "lon"])
    .compute()
)

print("Calculate GOES16 quantiles")
if os.path.exists(fn) and overwrite is False:
    q_ABI = xr.open_dataset(fn)
else:
    sat_input_filename_fmt = cfg.OBS.SATELLITES["GOES16"].CH13.filename_fmt
    ds_goes16_local = xr.open_mfdataset(
        sat_input_filename_fmt.replace("%Y%m%d_%H%M", "2020*"), parallel=True
    )
    q_ABI = ds_goes16_local.quantile(
        threshold_discard_percentile / 100, dim=["lat", "lon"]
    ).compute()
    q_ABI.to_netcdf(fn)
q_ABI = q_ABI.sel(time=slice("2020-01-08", "2020-02-18"))

print("High cloud occurrance")
count_highcloud_occurrance_DOM01 = (
    (q_DOM01 < threshold_discard_temperature).resample(time="1D").sum()
)
days_without_highclouds_DOM01 = count_highcloud_occurrance_DOM01.where(
    count_highcloud_occurrance_DOM01 == 0, drop=True
).time
count_highcloud_occurrance_DOM02 = (
    (q_DOM02 < threshold_discard_temperature).resample(time="1D").sum()
)
days_without_highclouds_DOM02 = count_highcloud_occurrance_DOM02.where(
    count_highcloud_occurrance_DOM02 == 0, drop=True
).time
count_highcloud_occurrance_CERES = (
    (q_ABI.C13 < threshold_discard_temperature).resample(time="1D").sum()
)
days_without_highclouds_CERES = count_highcloud_occurrance_CERES.where(
    count_highcloud_occurrance_CERES == 0, drop=True
).time

print("Write output")
DOM02_CERES_common_days_without_highclouds = list(
    set(days_without_highclouds_CERES.values).intersection(
        days_without_highclouds_DOM02.values
    )
)

pd.DataFrame(
    DOM02_CERES_common_days_without_highclouds, columns=["no_high_cloud"]
).to_parquet("../../data/result/no_high_clouds_DOM02.pq")

DOM01_CERES_common_days_without_highclouds = list(
    set(days_without_highclouds_CERES.values).intersection(
        days_without_highclouds_DOM01.values
    )
)
pd.DataFrame(
    DOM01_CERES_common_days_without_highclouds, columns=["no_high_cloud"]
).to_parquet("../../data/result/no_high_clouds_DOM01.pq")
