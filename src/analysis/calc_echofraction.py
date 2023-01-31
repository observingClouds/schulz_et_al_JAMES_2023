# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: covariability
#     language: python
#     name: covariability
# ---

# +
# Comparison of synthetitic and actual echo fraction by group e.g. meso-scale pattern

import logging
import os
import sys

import dask
import matplotlib
import matplotlib.pyplot as plt
import mpld3
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from intake import open_catalog
from matplotlib.dates import DateFormatter, HourLocator
from omegaconf import OmegaConf

sys.path.append("../helpers/")
import cluster_helpers as ch  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.FileHandler("../../logs/fig17.log"), logging.StreamHandler()],
)

if __name__ == "__main__":  # noqa: C901
    cfg = OmegaConf.load("../config/paths.cfg")
    # -

    client = ch.setup_cluster("local cluster", verbose=logging.ERROR)
    print(client)

    cat_address = (
        "https://raw.githubusercontent.com/eurec4a/eurec4a-intake/master/catalog.yml"
    )
    cat = open_catalog(cat_address)

    met = (
        cat.simulations.ICON.LES_CampaignDomain_control.meteogram_c_center_DOM02.to_dask()
    )

    conf_dict = {
        "DOM01": {"label": "ICON 624m", "color": "#109AFA"},
        "DOM02": {"label": "ICON 312m", "color": "red"},
        "obs": {"label": "GOES-16 ABI", "color": "black"},
    }

    fig_output_folder = cfg.ANALYSIS.MESOSCALE.PROFILES.dir_figures
    data_output_fmt = cfg.ANALYSIS.MESOSCALE.PROFILES.data_mean
    radar_fmt = cfg.OBS.BCO.Ka_radar_lev1_fmt
    resampling = "1D"
    overwrite = True
    fn_mean_out = "../../data/result/profile_means.nc"
    fn_stderr_out = "../../data/result/profile_stderr_means.nc"

    # +
    # Daily means
    resampling_means = {}

    station = "BCO"

    ## Simulations
    for experiment in ["control"]:  # ,'highCCN']:
        for domain in [1, 2]:
            print(domain)
            for threshold in [-50]:
                pamtra_ds = cat.simulations.ICON[f"LES_CampaignDomain_{experiment}"][
                    f"synthetic_radar_{station}_DOM0{domain}"
                ].to_dask()
                resampling_profile = (pamtra_ds.Z_att > threshold).sortby(
                    "time"
                ).resample(time=resampling).mean() * 100
                resampling_means[
                    experiment, domain, threshold
                ] = resampling_profile.compute()

    # -

    def count_non_NaN(x, **kwargs):
        """
        >>> da = xr.DataArray(
            np.linspace(0, 11, num=12),
            coords=[
                pd.date_range(
                    "1999-12-15",
                    periods=12,
                    freq=pd.DateOffset(months=1),
                )
            ],
            dims="time",
        )
        >>> da
        >>> da.rolling(time=3, center=True, min_periods=1).reduce(count_non_NaN)
        <xarray.DataArray (time: 12)>
        array([2., 3., 3., 2., 2., 2., 3., 3., 3., 3., 3., 2.])
        Coordinates:
        * time     (time) datetime64[ns] 1999-12-15 2000-01-15 ... 2000-11-15'
        """

        return np.sum(~np.isnan(x), **kwargs)

    # +
    # Daily means
    std_variability = 60 * 4
    resampling_std_means = {}

    ## Simulations
    for experiment in ["control"]:  # ,'highCCN']:
        for domain in [1, 2]:
            logging.info(domain)
            for threshold in [-50]:
                pamtra_ds = cat.simulations.ICON[f"LES_CampaignDomain_{experiment}"][
                    f"synthetic_radar_{station}_DOM0{domain}"
                ].to_dask()
                resampled_1T = (
                    (pamtra_ds.Z_att > threshold)
                    .sortby("time")
                    .resample(time="1T")
                    .nearest(tolerance="1T")
                )
                rolling = resampled_1T.rolling(
                    time=std_variability, center=True, min_periods=1
                )
                resampled_std_var = rolling.std(ddof=1)
                resampled_N = rolling.reduce(count_non_NaN)
                resampling_profile = (resampled_std_var / np.sqrt(resampled_N)).resample(
                    time=resampling
                ).mean() * 100
                resampling_std_means[
                    experiment, domain, threshold
                ] = resampling_profile.compute()
    # -

    ## Observations
    dates = pd.date_range("2020-01-12", "2020-02-20")

    radar_ds = cat.barbados.bco.radar_reflectivity.to_dask()
    radar_ds = (
        radar_ds.sel(time=slice(dates[0], dates[-1])).sel(range=slice(0, 500)).load()
    )

    minimum_measurements_per_day = 42160
    obs_per_date = radar_ds.time.groupby(radar_ds.time.dt.date).count()
    radar_ds_groups = []
    for _, grp in radar_ds.groupby(radar_ds.time.dt.date):
        if len(grp.time) >= minimum_measurements_per_day:
            radar_ds_groups.append(grp)
    radar_ds_sel = xr.concat(radar_ds_groups, dim="time")

    logging.info("Creating resampled means")
    for threshold in [-50]:
        resampling_profile = (radar_ds.Ze > threshold).resample(
            time=resampling
        ).mean() * 100
        resampling_means["Ka-Band", threshold] = resampling_profile.compute()

    # + tags=[]
    logging.info("Rolling mean std error calculation")
    for threshold in [-50]:
        resampled_2S = (
            (radar_ds.Ze > threshold).resample(time="2s").nearest(tolerance="1s")
        )
        rolling = resampled_2S.rolling(time=std_variability, center=True, min_periods=1)
        resampled_std_var = rolling.std(ddof=1)
        resampled_N = rolling.reduce(count_non_NaN)
        resampling_profile = (resampled_std_var / np.sqrt(resampled_N)).resample(
            time=resampling
        ).mean() * 100
        resampling_std_means["Ka-Band", threshold] = resampling_profile.compute()

    # +
    # Create common dataset
    logging.info("Calculate mean profiles")
    if os.path.exists(fn_mean_out) and overwrite is False:
        ds_mean = xr.open_dataset(fn_mean_out)
    else:
        ds_out = xr.Dataset()
        dss = []
        heights = resampling_means["control", 1, -50].height
        for key, val in resampling_means.items():
            if key[0] == "Ka-Band":
                da = (
                    val.expand_dims(
                        {"experiment": [key[0]], "dom": [0], "threshold": [key[1]]}
                    )
                    .rename({"range": "height"})
                    .reindex(height=heights, method="nearest")
                )
            else:
                da = val.expand_dims(
                    {"experiment": [key[0]], "dom": [key[1]], "threshold": [key[2]]}
                ).reindex(height=heights, method="nearest")
            da.name = "cf"
            dss.append(da)
        ds_mean = xr.merge(dss)
        ds_mean.to_netcdf(fn_mean_out)

    logging.info("Calculate std error of profiles")
    if os.path.exists(fn_stderr_out) and overwrite is False:
        ds_err = xr.open_dataset(fn_stderr_out)
    else:
        ds_out = xr.Dataset()
        dss = []
        heights = resampling_std_means["control", 1, -50].height
        for key, val in resampling_std_means.items():
            if key[0] == "Ka-Band":
                da = (
                    val.expand_dims(
                        {"experiment": [key[0]], "dom": [0], "threshold": [key[1]]}
                    )
                    .rename({"range": "height"})
                    .reindex(height=heights, method="nearest")
                )
            else:
                da = val.expand_dims(
                    {"experiment": [key[0]], "dom": [key[1]], "threshold": [key[2]]}
                ).reindex(height=heights, method="nearest")
            da.name = "cf"
            dss.append(da)
        ds_err = xr.merge(dss)
        ds_err.to_netcdf(fn_stderr_out)
