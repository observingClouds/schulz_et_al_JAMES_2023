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
import os
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
import plot_helpers as ph  # noqa: E402

cfg = OmegaConf.load("../config/paths.cfg")
params = OmegaConf.load("../config/mesoscale_params.yaml")

valid_cell_limit = 0.9

logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.FileHandler("../logs/fig14.log"), logging.StreamHandler()],
)

if __name__ == "__main__":
    client = ch.setup_cluster("local cluster", verbose=logging.ERROR)
    print(client)

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
    cat = open_catalog(
        "https://raw.githubusercontent.com/observingClouds/eurec4a-intake/simulations/catalog.yml"
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

    df_goes16_dom1

    alphas = {"exp1": 0.5, "exp2": 1}
    descr = {1: "high CCN", 2: "control"}

    # plt.figure(figsize=(8,2),dpi=300)
    df_simulation = xr.open_dataset(
        cfg.ANALYSIS.MESOSCALE.METRICS.output_filename_fmt.format(
            type="rttov", DOM=1, exp=2
        )
    )
    df_simulation = df_simulation.isel(time=df_simulation.percentile_BT > 100)
    times_sim = set(df_simulation.time.values)
    times_obs = set(df_goes16_dom1.time.values)
    common_times = sorted(times_obs.intersection(times_sim))
    data_obs = df_goes16_dom1.sel(time=slice("2020-01-10", np.max(list(common_times))))

    obs_1D_mean = data_obs.cloud_fraction.resample(time="1D").mean().compute()

    df_simulation = xr.open_dataset(
        cfg.ANALYSIS.MESOSCALE.METRICS.output_filename_fmt.format(
            type="rttov", DOM=2, exp=2
        )
    )
    DOM02_1D_mean = df_simulation.cloud_fraction.resample(time="1D").mean().compute()

    times = list(set(obs_1D_mean.time.values).intersection(DOM02_1D_mean.time.values))

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
    # -

    # Get datasets
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
    df_goes16_dom3 = df_goes16_dom2
    df_rttov_dom1 = xr.open_dataset(
        cfg.ANALYSIS.MESOSCALE.METRICS.output_filename_fmt.format(
            type="rttov", DOM=1, exp=2
        )
    )
    df_rttov_dom2 = xr.open_dataset(
        cfg.ANALYSIS.MESOSCALE.METRICS.output_filename_fmt.format(
            type="rttov", DOM=2, exp=2
        )
    )
    df_rttov_dom3 = xr.open_dataset(
        cfg.ANALYSIS.MESOSCALE.METRICS.output_filename_fmt.format(
            type="rttov", DOM=3, exp=2
        )
    )

    # Get days with sufficient measurements/output
    df_goes16_dom1_minsamples = 45  # 48 is optimum
    df_goes16_dom2_minsamples = 45  # 48 is optimum
    df_goes16_dom3_minsamples = 45  # 48 is optimum
    df_rttov_dom1_minsamples = 144  # 144 is optimum
    df_rttov_dom2_minsamples = 144  # 144 is optimum
    df_rttov_dom3_minsamples = 144  # 144 is optimum

    # Get available samples per day
    df_goes16_dom1_samples = (
        df_goes16_dom1.groupby(df_goes16_dom1.time.dt.date)
        .count()
        .reindex(date=df_goes16_dom1.time.values, method="ffill")
        .rename({"date": "time"})
    )
    df_goes16_dom2_samples = (
        df_goes16_dom2.groupby(df_goes16_dom2.time.dt.date)
        .count()
        .reindex(date=df_goes16_dom2.time.values, method="ffill")
        .rename({"date": "time"})
    )
    df_rttov_dom1_samples = (
        df_rttov_dom1.groupby(df_rttov_dom1.time.dt.date)
        .count()
        .reindex(date=df_rttov_dom1.time.values, method="ffill")
        .rename({"date": "time"})
    )
    df_rttov_dom2_samples = (
        df_rttov_dom2.groupby(df_rttov_dom2.time.dt.date)
        .count()
        .reindex(date=df_rttov_dom2.time.values, method="ffill")
        .rename({"date": "time"})
    )
    df_rttov_dom3_samples = (
        df_rttov_dom3.groupby(df_rttov_dom3.time.dt.date)
        .count()
        .reindex(date=df_rttov_dom3.time.values, method="ffill")
        .rename({"date": "time"})
    )

    # Restrict datasets to days with sufficient samples
    df_goes16_dom1 = df_goes16_dom1.sel(
        time=(df_goes16_dom1_samples >= df_goes16_dom1_minsamples).cloud_fraction
    )
    df_goes16_dom2 = df_goes16_dom2.sel(
        time=(df_goes16_dom2_samples >= df_goes16_dom2_minsamples).cloud_fraction
    )
    df_rttov_dom1 = df_rttov_dom1.sel(
        time=(df_rttov_dom1_samples >= df_rttov_dom1_minsamples).cloud_fraction
    )
    df_rttov_dom2 = df_rttov_dom2.sel(
        time=(df_rttov_dom2_samples >= df_rttov_dom2_minsamples).cloud_fraction
    )
    df_rttov_dom3 = df_rttov_dom3.sel(
        time=(df_rttov_dom3_samples >= df_rttov_dom3_minsamples).cloud_fraction
    )

    # # Restrict datasets to common days
    obs_common = set(df_goes16_dom1.time.dt.date.values).intersection(
        df_goes16_dom2.time.dt.date.values
    )
    sim_common = (
        set(df_rttov_dom1.time.dt.date.values)
        .intersection(df_rttov_dom2.time.dt.date.values)
        .intersection(df_rttov_dom3.time.dt.date.values)
    )

    # Restrict datasets to days without high clouds
    ## percentile method
    valid_cell_limit
    high_cloud_flag_goes16_dom1 = (
        ~(df_goes16_dom1.valid_cells < valid_cell_limit)
        .groupby(df_goes16_dom1.time.dt.date)
        .any()
    )
    high_cloud_flag_goes16_dom2 = (
        ~(df_goes16_dom2.valid_cells < valid_cell_limit)
        .groupby(df_goes16_dom2.time.dt.date)
        .any()
    )
    high_cloud_flag_rttov_dom1 = (
        ~(df_rttov_dom1.valid_cells < valid_cell_limit)
        .groupby(df_rttov_dom1.time.dt.date)
        .any()
    )
    high_cloud_flag_rttov_dom2 = (
        ~(df_rttov_dom2.valid_cells < valid_cell_limit)
        .groupby(df_rttov_dom2.time.dt.date)
        .any()
    )
    high_cloud_flag_rttov_dom3 = (
        ~(df_rttov_dom3.valid_cells < valid_cell_limit)
        .groupby(df_rttov_dom3.time.dt.date)
        .any()
    )

    common_times = (
        high_cloud_flag_goes16_dom1
        & high_cloud_flag_goes16_dom2
        & high_cloud_flag_rttov_dom1
        & high_cloud_flag_rttov_dom2
    )

    #    df_goes16_dom1 = df_goes16_dom1.sel(
    #        time=common_times.sel(date=df_goes16_dom1.time.dt.date)
    #    )
    #    df_goes16_dom2 = df_goes16_dom2.sel(
    #        time=common_times.sel(date=df_goes16_dom2.time.dt.date)
    #    )
    #    df_rttov_dom1 = df_rttov_dom1.sel(
    #        time=common_times.sel(date=df_rttov_dom1.time.dt.date)
    #    )
    #    df_rttov_dom2 = df_rttov_dom2.sel(
    #        time=common_times.sel(date=df_rttov_dom2.time.dt.date)
    #    )
    #    df_rttov_dom3 = df_rttov_dom3.sel(
    #        time=common_times.sel(date=df_rttov_dom3.time.dt.date)
    #    )

    common_times = obs_common.intersection(sim_common)
    # exclude Feb 2 to test if the prolonged peak in OBS is caused by it
    # common_times = common_times.difference({dt.datetime(2020,2,5).date()})
    # common_times = common_times.difference({dt.datetime(2020,2,5).date(),
    #                                         dt.datetime(2020,2,4).date(),
    #                                         dt.datetime(2020,2,2).date()})
    common_times = list(common_times)

    df_goes16_dom1 = df_goes16_dom1.sel(
        time=np.in1d(df_goes16_dom1.time.dt.date, common_times)
    )
    df_goes16_dom2 = df_goes16_dom2.sel(
        time=np.in1d(df_goes16_dom2.time.dt.date, common_times)
    )
    df_rttov_dom1 = df_rttov_dom1.sel(
        time=np.in1d(df_rttov_dom1.time.dt.date, common_times)
    )
    df_rttov_dom2 = df_rttov_dom2.sel(
        time=np.in1d(df_rttov_dom2.time.dt.date, common_times)
    )
    df_rttov_dom3 = df_rttov_dom3.sel(
        time=np.in1d(df_rttov_dom3.time.dt.date, common_times)
    )

    logging.info(np.unique(df_rttov_dom1.time.dt.date))
    # Print overall means
    logging.info("Overall cloud fraction without days of high clouds")
    logging.info(f"OBS (DOM01): {df_goes16_dom1.cloud_fraction.mean().item(0)*100:.2f}")
    logging.info(f"OBS (DOM02): {df_goes16_dom2.cloud_fraction.mean().item(0)*100:.2f}")
    logging.info(f"ICON (DOM01): {df_rttov_dom1.cloud_fraction.mean().item(0)*100:.2f}")
    logging.info(f"ICON (DOM02): {df_rttov_dom2.cloud_fraction.mean().item(0)*100:.2f}")
    logging.info(f"ICON (DOM03): {df_rttov_dom3.cloud_fraction.mean().item(0)*100:.2f}")

    logging.info("Overall cloud fraction without days of high clouds")
    logging.info(
        f"OBS (DOM01): {df_goes16_dom1.cloud_fraction.median().item(0)*100:.2f}"
    )
    logging.info(
        f"OBS (DOM02): {df_goes16_dom2.cloud_fraction.median().item(0)*100:.2f}"
    )
    logging.info(
        f"ICON (DOM01): {df_rttov_dom1.cloud_fraction.median().item(0)*100:.2f}"
    )
    logging.info(
        f"ICON (DOM02): {df_rttov_dom2.cloud_fraction.median().item(0)*100:.2f}"
    )
    logging.info(
        f"ICON (DOM03): {df_rttov_dom3.cloud_fraction.median().item(0)*100:.2f}"
    )

    # Calc daily means
    daily_mean_goes16_dom1 = (
        df_goes16_dom1.groupby(df_goes16_dom1.time.dt.date)
        .mean()
        .reindex(date=df_goes16_dom1.time.values, method="ffill")
        .rename({"date": "time"})
    )
    daily_mean_goes16_dom2 = (
        df_goes16_dom2.groupby(df_goes16_dom2.time.dt.date)
        .mean()
        .reindex(date=df_goes16_dom2.time.values, method="ffill")
        .rename({"date": "time"})
    )
    daily_mean_rttov_dom1 = (
        df_rttov_dom1.groupby(df_rttov_dom1.time.dt.date)
        .mean()
        .reindex(date=df_rttov_dom1.time.values, method="ffill")
        .rename({"date": "time"})
    )
    daily_mean_rttov_dom2 = (
        df_rttov_dom2.groupby(df_rttov_dom2.time.dt.date)
        .mean()
        .reindex(date=df_rttov_dom2.time.values, method="ffill")
        .rename({"date": "time"})
    )
    daily_mean_rttov_dom3 = (
        df_rttov_dom3.groupby(df_rttov_dom3.time.dt.date)
        .mean()
        .reindex(date=df_rttov_dom3.time.values, method="ffill")
        .rename({"date": "time"})
    )

    # Calculate daily anomaly
    daily_anomaly_goes16_dom1 = df_goes16_dom1 - daily_mean_goes16_dom1
    daily_anomaly_goes16_dom2 = df_goes16_dom2 - daily_mean_goes16_dom2
    daily_anomaly_rttov_dom1 = df_rttov_dom1 - daily_mean_rttov_dom1
    daily_anomaly_rttov_dom2 = df_rttov_dom2 - daily_mean_rttov_dom2
    daily_anomaly_rttov_dom3 = df_rttov_dom3 - daily_mean_rttov_dom3

    df_all = daily_anomaly_rttov_dom2.cloud_fraction.to_dataframe(name="cloud fraction")
    df_all["product"] = "ICON-312m"

    # +
    df = daily_anomaly_rttov_dom1.cloud_fraction.to_dataframe(name="cloud fraction")
    df["product"] = "ICON-624m"
    df_all = pd.concat([df_all, df])

    df = daily_anomaly_rttov_dom3.cloud_fraction.to_dataframe(name="cloud fraction")
    df["product"] = "ICON-156m"
    df_all = pd.concat([df_all, df])

    df = daily_anomaly_goes16_dom1.cloud_fraction.to_dataframe(name="cloud fraction")
    df["product"] = "GOES16"
    df_all = pd.concat([df_all, df])
    # -

    df_all["hour"] = df_all.index.hour

    # +
    # sns.set_context('paper')
    fig = plt.figure(figsize=(3, 2), dpi=150)
    products_to_show = ["GOES16", "ICON-312m", "ICON-624m", "ICON-156m"]
    palette = {
        "ICON-156m": f"{conf_dict['DOM{DOM:02g}'.format(DOM=3)]['color']}",
        "ICON-312m": f"{conf_dict['DOM{DOM:02g}'.format(DOM=2)]['color']}",
        "ICON-624m": f"{conf_dict['DOM{DOM:02g}'.format(DOM=1)]['color']}",
        "GOES16": "grey",
    }
    linewidth = 3

    sns.boxplot(
        data=df_all,
        x="hour",
        hue="product",
        hue_order=products_to_show,
        y="cloud fraction",
        showfliers=False,
        whis=0,
        palette=palette,
        linewidth=0,
    )
    median_goes16 = daily_anomaly_goes16_dom1.cloud_fraction.groupby(
        daily_anomaly_goes16_dom1.time.dt.hour
    ).median()
    median_goes16.plot(color="k", linewidth=linewidth)
    median_624 = daily_anomaly_rttov_dom1.cloud_fraction.groupby(
        daily_anomaly_rttov_dom1.time.dt.hour
    ).median()
    median_156 = daily_anomaly_rttov_dom3.cloud_fraction.groupby(
        daily_anomaly_rttov_dom3.time.dt.hour
    ).median()
    if "ICON-624m" in products_to_show:
        median_624.plot(
            color=conf_dict["DOM{DOM:02g}".format(DOM=1)]["color"], linewidth=linewidth
        )
    if "ICON-156m" in products_to_show:
        median_156.plot(
            color=conf_dict["DOM{DOM:02g}".format(DOM=3)]["color"], linewidth=linewidth
        )
    median_312 = daily_anomaly_rttov_dom2.cloud_fraction.groupby(
        daily_anomaly_rttov_dom1.time.dt.hour
    ).median()
    median_312.plot(
        color=conf_dict["DOM{DOM:02g}".format(DOM=2)]["color"], linewidth=linewidth
    )
    plt.xlabel("hour / LT")
    plt.ylabel(r"$C_{\mathrm{B}}'$")
    plt.xticks(ticks=[0, 4, 8, 12, 16, 20], labels=["20", "0", "4", "8", "12", "16"])
    plt.yticks(ticks=[-0.06, -0.03, 0, 0.03, 0.06])
    plt.legend(ncol=2)
    plt.gca().get_legend().remove()
    sns.despine(offset=10)
    out_fn = f"FIG_CC_diurnalcycle_anomaly_{'_'.join(products_to_show)}.pdf"
    out_path = os.path.join(fig_output_folder, out_fn)
    out_dir = os.path.dirname(out_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plt.savefig(out_path, bbox_inches="tight")

    # +
    logging.info("Maximum")
    logging.info(median_goes16.hour.isel(hour=np.argmax(median_goes16.values)).item(0))
    logging.info(median_156.hour.isel(hour=np.argmax(median_156.values)).item(0))
    logging.info(median_312.hour.isel(hour=np.argmax(median_312.values)).item(0))
    logging.info(median_624.hour.isel(hour=np.argmax(median_624.values)).item(0))

    logging.info("Minimum")
    logging.info(median_goes16.hour.isel(hour=np.argmin(median_goes16.values)).item(0))
    logging.info(median_156.hour.isel(hour=np.argmin(median_156.values)).item(0))
    logging.info(median_312.hour.isel(hour=np.argmin(median_312.values)).item(0))
    logging.info(median_624.hour.isel(hour=np.argmin(median_624.values)).item(0))

    logging.info("Amplitude")
    logging.info(median_goes16.max() - median_goes16.min())
    logging.info(median_156.max() - median_156.min())
    logging.info(median_312.max() - median_312.min())
    logging.info(median_624.max() - median_624.min())
