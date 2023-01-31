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

# works with xarray version 2012.10.0, but not with 2012.11.0 and 2012.12.0

import logging
import sys

import matplotlib
import matplotlib.dates as md
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import xarray as xr
from intake import open_catalog
from omegaconf import OmegaConf

sys.path.append("../src/helpers/")

import cluster_helpers as ch  # noqa: E402

if __name__ == "__main__":
    cfg = OmegaConf.load("../config/paths.cfg")
    params = OmegaConf.load("../config/mesoscale_params.yaml")

    df_goes16_dom1 = xr.open_dataset(
        cfg.ANALYSIS.MESOSCALE.METRICS.output_filename_fmt.format(
            type="goes16", DOM=1, exp="_"
        )[:-4]
        + ".nc"
    )
    df_goes16_dom2 = xr.open_dataset(
        cfg.ANALYSIS.MESOSCALE.METRICS.output_filename_fmt.format(
            type="goes16", DOM=2, exp="_"
        )[:-4]
        + ".nc"
    )
    df_rttov_dom1 = xr.open_dataset(
        cfg.ANALYSIS.MESOSCALE.METRICS.output_filename_fmt.format(
            type="rttov", DOM=2, exp=2
        )
    )
    df_rttov_dom2 = xr.open_dataset(
        cfg.ANALYSIS.MESOSCALE.METRICS.output_filename_fmt.format(
            type="rttov", DOM=2, exp=2
        )
    )
    df_rttov_dom2 = xr.open_dataset(
        cfg.ANALYSIS.MESOSCALE.METRICS.output_filename_fmt.format(
            type="rttov", DOM=3, exp=2
        )
    )
    fig_output_folder = cfg.ANALYSIS.MESOSCALE.METRICS.dir_figures
    resampling = "10T"

    def replace_dim(da, olddim, newdim):
        renamed = da.rename({olddim: newdim.name})

        # note that alignment along a dimension is skipped when you are overriding
        # the relevant coordinate values
        renamed.coords[newdim.name] = newdim
        return renamed

    # -

    client = ch.setup_cluster("local cluster", verbose=logging.ERROR)
    print(client)

    cat = open_catalog(
        "https://raw.githubusercontent.com/eurec4a/eurec4a-intake/master/catalog.yml"
    )

    # +
    conf_dict = {
        "DOM01": {"label": "ICON 624m", "color": "#109AFA"},
        "DOM02": {"label": "ICON 312m", "color": "red"},
        "DOM03": {"label": "ICON 156m", "color": "brown"},
        "obs": {"label": "GOES-16 ABI", "color": "black"},
    }

    geobounds = {}
    geobounds["lat_min"] = params.metrics.geobounds.lat_min
    geobounds["lat_max"] = params.metrics.geobounds.lat_max
    geobounds["lon_min"] = params.metrics.geobounds.lon_min
    geobounds["lon_max"] = params.metrics.geobounds.lon_max
    # -

    alphas = {"exp1": 0.5, "exp2": 1}
    descr = {1: "high CCN", 2: "control"}

    df_simulation = xr.open_dataset(
        cfg.ANALYSIS.MESOSCALE.METRICS.output_filename_fmt.format(
            type="rttov", DOM=1, exp=2
        )
    )
    df_simulation = df_simulation.isel(index=df_simulation.percentile_BT > 100)
    times_sim = set(df_simulation.time.values)
    times_obs = set(df_goes16_dom1.time.values)
    common_times = sorted(times_obs.intersection(times_sim))
    data_obs = df_goes16_dom1.set_index(index="time").sel(
        index=slice("2020-01-10", np.max(list(common_times)))
    )

    obs_1D_mean = data_obs.cloud_fraction.resample(index="1D").mean().compute()

    df_simulation = xr.open_dataset(
        cfg.ANALYSIS.MESOSCALE.METRICS.output_filename_fmt.format(
            type="rttov", DOM=2, exp=2
        )
    )
    df_simulation = df_simulation.set_index(index="time")
    DOM02_1D_mean = df_simulation.cloud_fraction.resample(index="1D").mean().compute()

    times = list(set(obs_1D_mean.index.values).intersection(DOM02_1D_mean.index.values))

    df_nohighClouds = pd.read_parquet("../data/result/no_high_clouds_DOM02.pq")

    # +
    ds_max = xr.open_dataset("../data/result/max_pattern_freq.nc")

    max_freq = ds_max["max_freq"]
    max_pattern = ds_max["max_pattern"]
    mean_pattern_freq = ds_max["mean_freq"]
    threshold_freq = params.manual_classifications.threshold_pattern
    color_dict = {
        "Sugar": "#A1D791",
        "Flower": "#93D2E2",
        "Fish": "#2281BB",
        "Gravel": "#3EAE47",
        "Flowers": "#93D2E2",
        "Unclassified": "grey",
    }

    # +
    fig, axs = plt.subplots(1, 1)
    data_1_nohigh = (
        DOM02_1D_mean.sel(index=times).sel(index=list(df_nohighClouds["no_high_cloud"]))
        * 100
    )
    data_2_nohigh = (
        obs_1D_mean.sel(index=times).sel(index=list(df_nohighClouds["no_high_cloud"]))
        * 100
    )

    data_1_withhigh = DOM02_1D_mean.sel(index=times) * 100
    data_2_withhigh = obs_1D_mean.sel(index=times) * 100

    colors = []
    for date in data_1_nohigh.index.values:
        if max_freq.sel(date=date) > threshold_freq:
            color = color_dict[
                mean_pattern_freq.pattern.values[max_pattern.sel(date=date)]
            ]
            colors.append(color)
        else:
            colors.append("grey")

    axs.scatter(data_1_nohigh, data_2_nohigh, color=colors, s=30, zorder=100)
    colors = []
    for date in data_1_withhigh.index.values:
        if max_freq.sel(date=date) > threshold_freq:
            color = color_dict[
                mean_pattern_freq.pattern.values[max_pattern.sel(date=date)]
            ]
            colors.append(color)
        else:
            colors.append("grey")
    axs.scatter(
        data_1_withhigh, data_2_withhigh, color=colors, s=30, zorder=1, alpha=0.2
    )
    # sns.regplot(obs_1D_mean.sel(index=times), DOM02_1D_mean.sel(index=times))
    slope, intercept, _, _, _ = scipy.stats.linregress(x=data_1_nohigh, y=data_2_nohigh)

    def regression_func(x):
        return slope * x + intercept

    axs.plot(
        [np.min(data_1_nohigh), np.max(data_1_nohigh)],
        [regression_func(f) for f in (np.min(data_1_nohigh), np.max(data_1_nohigh))],
        color="black",
    )
    axs.text(
        14,
        6,
        r"$C_{\mathrm{B}}\mathrm{(OBS)}}$"
        + f"={slope:.2f}"
        + r"$\cdot C_{\mathrm{B}}\mathrm{(SIM)}}$"
        + f"+{intercept:.2f}",
    )

    slope, intercept, _, _, _ = scipy.stats.linregress(
        x=data_1_withhigh, y=data_2_withhigh
    )
    axs.plot(
        [np.min(data_1_withhigh), np.max(data_1_withhigh)],
        [regression_func(f) for f in (np.min(data_1_withhigh), np.max(data_1_withhigh))],
        color="black",
        alpha=0.2,
    )
    axs.text(
        14,
        3,
        r"$C_{\mathrm{B}}\mathrm{(OBS)}}$"
        + f"={slope:.2f}"
        + r"$\cdot C_{\mathrm{B}}\mathrm{(SIM)}}$"
        + f"+{intercept:.2f}",
        alpha=0.3,
    )

    axs.set_ylabel(r"$C_{\mathrm{B}}\mathrm{(OBS)}}$ / %")
    axs.set_xlabel(r"$C_{\mathrm{B}}\mathrm{(SIM)}}$ / %")
    axs.set_yticks(np.arange(5, 45, 10))
    axs.set_xticks(np.arange(5, 45, 10))
    axs.plot(
        [np.min(data_2_nohigh), np.max(data_2_nohigh)],
        [np.min(data_2_nohigh), np.max(data_2_nohigh)],
        color="grey",
        linestyle="--",
    )
    sns.despine()
    axs.set_aspect(1)
    plt.savefig(
        "../figures/cloud_cover_scatter_nohighclouds_and_highclouds.pdf",
        bbox_inches="tight",
    )
    # -

    print("Statistics without high clouds")
    print(
        "DOM02 mean:",
        (
            DOM02_1D_mean.sel(index=times).sel(
                index=list(df_nohighClouds["no_high_cloud"])
            )
            * 100
        )
        .mean()
        .item(0),
    )
    print(
        "DOM02 median:",
        (
            DOM02_1D_mean.sel(index=times).sel(
                index=list(df_nohighClouds["no_high_cloud"])
            )
            * 100
        )
        .median()
        .item(0),
    )
    print(
        "OBS mean:",
        (
            obs_1D_mean.sel(index=times).sel(
                index=list(df_nohighClouds["no_high_cloud"])
            )
            * 100
        )
        .mean()
        .item(0),
    )
    print(
        "OBS median:",
        (
            obs_1D_mean.sel(index=times).sel(
                index=list(df_nohighClouds["no_high_cloud"])
            )
            * 100
        )
        .median()
        .item(0),
    )
    print(f"based on {len(times)} days")
