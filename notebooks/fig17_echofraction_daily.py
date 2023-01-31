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

import datetime as dt
import logging

import matplotlib
import matplotlib.pyplot as plt
import mpld3
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib.dates import DateFormatter, HourLocator

logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.FileHandler("../logs/fig17.log"), logging.StreamHandler()],
)

if __name__ == "__main__":  # noqa: C901
    # -

    conf_dict = {
        "DOM01": {"label": "ICON 624m", "color": "#109AFA"},
        "DOM02": {"label": "ICON 312m", "color": "red"},
        "obs": {"label": "GOES-16 ABI", "color": "black"},
    }

    # +
    dates = pd.date_range("2020-01-12", "2020-02-20")

    timeslice = slice("2020-01-12", "2020-02-20")
    ds_err = xr.open_dataset("../data/result/profile_stderr_means.nc")
    ds_mean = xr.open_dataset("../data/result/profile_means.nc")

    fig, ax = plt.subplots(figsize=(8, 3), dpi=300)
    dfs = []
    for setup in [
        {
            "selector": {
                "height": 300,
                "threshold": -50,
                "experiment": "control",
                "dom": 2,
            },
            "color": "orange",
            "label": "ICON 312m",
        },
        {
            "selector": {
                "height": 300,
                "threshold": -50,
                "experiment": "Ka-Band",
                "dom": 0,
            },
            "color": "grey",
            "label": "OBS",
        },
    ]:
        selector = setup["selector"]
        data = (
            ds_mean.cf.sel({"height": selector["height"]}, method="nearest")
            .sel({k: selector[k] for k in ["threshold", "experiment", "dom"]})
            .sel(time=timeslice)
        )
        anomaly = (data - data.mean(dim="time")).to_series()
        anomaly.name = setup["label"]
        dfs.append(anomaly)

    df = pd.concat(dfs, axis=1)
    plt.bar(
        df.index - dt.timedelta(hours=3),
        df["ICON 312m"],
        label="ICON 312m",
        color="orange",
        width=dt.timedelta(hours=6),
    )
    plt.bar(
        df.index + dt.timedelta(hours=3),
        df["OBS"],
        label="OBS",
        color="grey",
        width=dt.timedelta(hours=6),
    )
    ax.xaxis.set_major_locator(HourLocator(interval=72))
    ax.xaxis.set_major_formatter(DateFormatter("%d.%m."))
    plt.legend(bbox_to_anchor=(1.0, 0.8))
    plt.xlim(dt.datetime(2020, 1, 11), dt.datetime(2020, 2, 20))
    plt.xticks(rotation=45)
    plt.xlabel("")
    plt.ylabel(r"$C_\mathrm{E}'(300\mathrm{m})$ / %")
    plt.tight_layout()
    sns.despine()
    plt.savefig(
        "../figures/daily_echo_fraction_anomaly_timeseries@300m_ICON312m+OBS.pdf",
        bbox_inches="tight",
    )
    # -

    df_no_high_clouds = pd.read_parquet("../data/result/no_high_clouds_DOM02.pq")

    # +
    interactive = False
    high_clouds = True
    exclude_10012020 = False
    fig, ax = plt.subplots()
    for setup1, setup2 in [
        (
            {
                "selector": {
                    "height": 300,
                    "threshold": -50,
                    "experiment": "control",
                    "dom": 2,
                }
            },
            {
                "selector": {
                    "height": 300,
                    "threshold": -50,
                    "experiment": "Ka-Band",
                    "dom": 0,
                },
                "label": "DOM02",
            },
        ),
    ]:
        selector1 = setup1["selector"]
        selector2 = setup2["selector"]
        data1 = ds_mean.cf.sel({"height": selector1["height"]}, method="nearest").sel(
            {k: selector1[k] for k in ["threshold", "experiment", "dom"]}
        )
        data2 = ds_mean.cf.sel({"height": selector2["height"]}, method="nearest").sel(
            {k: selector2[k] for k in ["threshold", "experiment", "dom"]}
        )
        data1_err = ds_err.cf.sel({"height": selector1["height"]}, method="nearest").sel(
            {k: selector1[k] for k in ["threshold", "experiment", "dom"]}
        )
        data2_err = ds_err.cf.sel({"height": selector2["height"]}, method="nearest").sel(
            {k: selector2[k] for k in ["threshold", "experiment", "dom"]}
        )
        assert np.all(data1.time.values == data2.time.values), "Times do not agree"
        if exclude_10012020:
            data1 = data1.sel(time=slice("2020-01-11", None))
            data2 = data2.sel(time=slice("2020-01-11", None))
            data1_err = data1_err.sel(time=slice("2020-01-11", None))
            data2_err = data2_err.sel(time=slice("2020-01-11", None))
        if interactive:
            scatter = plt.scatter(
                data1 - data1.mean(dim="time"),
                data2 - data2.mean(dim="time"),
                marker=".",
                color="w",
                alpha=0,
                s=100,
            )
        mean1 = data1.mean(dim="time")
        mean2 = data2.mean(dim="time")
        ax.errorbar(
            data1 - mean1,
            data2 - mean2,
            xerr=data1_err,
            yerr=data2_err,
            label=setup2["label"],
            marker=".",
            linestyle="",
            color="grey",
            alpha=1,
        )
        if high_clouds:
            data1 = data1.sel(time=df_no_high_clouds.no_high_cloud.values)
            data2 = data2.sel(time=df_no_high_clouds.no_high_cloud.values)
            data1_err = data1_err.sel(time=df_no_high_clouds.no_high_cloud.values)
            data2_err = data2_err.sel(time=df_no_high_clouds.no_high_cloud.values)
            ax.errorbar(
                data1 - mean1,
                data2 - mean2,
                xerr=data1_err,
                yerr=data2_err,
                label=setup2["label"],
                marker=".",
                linestyle="",
                color="black",
                alpha=1,
            )
    if interactive:
        labels = []
        for i, time in enumerate(data1.time.dt.strftime("%d.%m.%Y").values):
            if not scatter._offsets.mask[:, 0][i]:  # ignoring masked labels
                labels.append("{0}".format(time))
        tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=labels)

    plt.xlabel(r"$C_\mathrm{E,ICON}'(300\mathrm{m})$ / %")
    plt.ylabel(r"$C_\mathrm{E,OBS}'(300\mathrm{m})$ / %")
    plt.plot([-15, 43], [-15, 43], color="lightgrey", linestyle="--")
    ax.set_ylim(-15, 43)
    ax.set_xlim(-15, 43)
    ax.set_aspect(1)
    # plt.legend(bbox_to_anchor=(1.2,0.8))
    sns.despine()
    if interactive:
        mpld3.plugins.connect(fig, tooltip)
        mpld3.display()
        mpld3.save_html(
            fig, "../figures/echofraction_scatter_obs_vs_dom02_with_err.html"
        )
    else:
        plt.savefig(
            "../figures/echofraction_scatter_obs_vs_dom02_with_err.pdf",
            bbox_inches="tight",
        )
