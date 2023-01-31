import logging
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr
from intake import open_catalog
from omegaconf import OmegaConf

sys.path.append("../helpers/")
import cluster_helpers as ch  # noqa: E402
import plot_helpers as ph  # noqa: E402

cfg = OmegaConf.load("../../config/paths.cfg")

if __name__ == "__main__":
    client = ch.setup_cluster("local cluster", verbose=logging.ERROR)
    print(client)

    cat_address = (
        "https://raw.githubusercontent.com/observingClouds/"
        "eurec4a-intake/ICON-LES-control-DOM03/catalog.yml"
    )
    cat = open_catalog(cat_address)

    fig_output_folder = cfg.ANALYSIS.MESOSCALE.PROFILES.dir_figures
    data_output_fmt = cfg.ANALYSIS.MESOSCALE.PROFILES.data_mean
    radar_fmt = cfg.OBS.BCO.Ka_radar_lev1_fmt
    resampling = "1D"
    station = "BCO"

    # Daily means
    resampling_means = {}

    ## Simulations
    for experiment in ["control", "highCCN"]:
        for domain in [1, 2, 3]:
            logging.debug(domain)
            try:
                pamtra_ds = cat.simulations.ICON[f"LES_CampaignDomain_{experiment}"][
                    f"synthetic_radar_{station}_DOM0{domain}"
                ].to_dask()
            except KeyError:
                continue
            resampling_profile = (pamtra_ds.Z_att > -50).sortby("time").resample(
                time=resampling
            ).mean() * 100
            resampling_means[experiment, domain] = resampling_profile.compute()

    ## Observations
    dates = pd.date_range("2020-01-10", "2020-02-18")
    radar_ds = cat.barbados.bco.radar_reflectivity.to_dask()

    minimum_measurements_per_day = 42160
    obs_per_date = radar_ds.time.groupby(radar_ds.time.dt.date).count()
    radar_ds_groups = []
    for _, grp in radar_ds.groupby(radar_ds.time.dt.date):
        if len(grp.time) >= minimum_measurements_per_day:
            radar_ds_groups.append(grp)
    radar_ds_sel = xr.concat(radar_ds_groups, dim="time")

    if os.path.exists(data_output_fmt.format(resampling + station)):
        ds_out = xr.open_dataset(data_output_fmt.format(resampling + station))
    else:
        # Create common dataset
        resampling_profile = (radar_ds.Zf > -50).resample(time=resampling).mean() * 100
        resampling_means["Ka-Band"] = resampling_profile.compute()

        ds_out = xr.Dataset()
        ds_out["CF_kaband"] = resampling_means["Ka-Band"]
        ds_out["control_CF_ICON_DOM01"] = resampling_means["control", 1]
        ds_out["control_CF_ICON_DOM02"] = (
            resampling_means["control", 2]
            .reindex(height=ds_out.height, method="nearest")
            .reindex(time=ds_out.time)
        )
        ds_out["control_CF_ICON_DOM03"] = (
            resampling_means["control", 3]
            .reindex(height=ds_out.height, method="nearest")
            .reindex(time=ds_out.time)
        )
        ds_out["highCCN_CF_ICON_DOM01"] = resampling_means["highCCN", 1]
        ds_out["highCCN_CF_ICON_DOM02"] = (
            resampling_means["highCCN", 2]
            .reindex(height=ds_out.height, method="nearest")
            .reindex(time=ds_out.time)
        )
        ds_out.to_netcdf(data_output_fmt.format(resampling + station))
