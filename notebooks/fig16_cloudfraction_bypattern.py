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
import numpy as np
import pandas as pd
import pint
import pint_xarray
import seaborn as sns
import xarray as xr
from intake import open_catalog
from omegaconf import OmegaConf
from scipy.stats import sem
from sklearn.utils import resample

sys.path.append("../src/helpers/")
sys.path.append("/home/m/m300408/GitProjects/Thermodynamics")
import aes_thermo as thermo  # noqa: E402
import cluster_helpers as ch  # noqa: E402
import plot_helpers as ph  # noqa: E402

cfg = OmegaConf.load("../config/paths.cfg")
# -

if __name__ == "__main__":  # noqa: C901
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.FileHandler("../logs/fig15.log"), logging.StreamHandler()],
    )

    client = ch.setup_cluster("local cluster", verbose=logging.ERROR)
    print(client)

    conf_dict = {
        "DOM01": {"label": "ICON 624m", "color": "#109AFA"},
        "DOM02": {"label": "ICON 312m", "color": "red"},
        "DOM03": {"label": "ICON 156m", "color": "brown"},
        "obs": {"label": "GOES-16 ABI", "color": "black"},
    }

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

    # +
    # Daily means
    resampling_means = {}

    ## Simulations
    for experiment in ["control", "highCCN"]:
        for domain in [1, 2, 3]:
            print(domain)
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
    # -

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

    # ## Read groups
    # ### Meso-scale patterns

    if os.path.exists(data_output_fmt.format(resampling + station)):
        ds_out = xr.open_dataset(data_output_fmt.format(resampling + station))
    else:
        logging.error("Please run `calc_vertical_echoprofile.py` first")

    output_mostcommon_classifications = (
        cfg.ANALYSIS.MESOSCALE.CLASSIFICATIONS.manual.IR.class_decision
    )

    df = pd.read_json(output_mostcommon_classifications)
    common_dates = sorted(
        set(df.index.floor("1D").values).intersection(ds_out.time.dt.floor("1D").values)
    )
    ds_out = ds_out.sel(time=common_dates)
    ds_out["pattern"] = xr.DataArray(
        df.loc[pd.to_datetime(common_dates, utc=True)].pattern.values,
        dims={"time": ds_out.time},
    )

    # Assuming observations are available, select only those that have model output
    ds_sel = ds_out.sel(time=~ds_out.control_CF_ICON_DOM02.isnull().any(dim="height"))

    mean = ds_sel.mean(dim="time")
    logging.info(
        f"OBS: max:{mean.CF_kaband.max().item(0)}, "
        + f"@{mean.range[np.argmax(mean.CF_kaband.values)]:.0f}m"
    )
    logging.info(
        f"Control DOM01: max:{mean.control_CF_ICON_DOM01.max().item(0)}, "
        + f"@{mean.height[np.argmax(mean.control_CF_ICON_DOM01.values)]:.0f}m"
    )
    logging.info(
        f"Control DOM02: max:{mean.control_CF_ICON_DOM02.max().item(0)}, "
        + f"@{mean.height[np.argmax(mean.control_CF_ICON_DOM02.values)]:.0f}m"
    )
    logging.info(
        f"Control DOM03: max:{mean.control_CF_ICON_DOM03.max().item(0)}, "
        + f"@{mean.height[np.argmax(mean.control_CF_ICON_DOM03.values)]:.0f}m"
    )
    logging.info(f'OBS: @2km:{mean.CF_kaband.sel(range=2000, method="nearest").item(0)}')
    logging.info(
        "Control DOM01:"
        + f' @2km:{mean.control_CF_ICON_DOM01.sel(height=2000, method="nearest").item(0)}'
    )
    logging.info(
        "Control DOM02:"
        + f' @2km:{mean.control_CF_ICON_DOM02.sel(height=2000, method="nearest").item(0)}'
    )
    logging.info(
        "Control DOM03:"
        + f' @2km:{mean.control_CF_ICON_DOM03.sel(height=2000, method="nearest").item(0)}'
    )

    # # Get inversion height

    # +
    inv_heights = {}
    for dom in [1, 2, 3]:
        print(dom)
        ds_mtrgm = cat.simulations.ICON.LES_CampaignDomain_control[
            f"meteogram_BCO_DOM0{dom}"
        ].to_dask()
        ds_mtrgm["height"] = (
            ds_mtrgm.height.pint.quantify() * pint.unit.Unit("km")
        ).pint.dequantify()
        ds_mtrgm["height_2"] = (
            (ds_mtrgm.height_2 / 1000).pint.quantify() * pint.unit.Unit("km")
        ).pint.dequantify()
        ds_mtrgm["qt"] = ds_mtrgm.QC + ds_mtrgm.QV + ds_mtrgm.QR
        ds_mtrgm["qt"] = (
            ds_mtrgm["qt"].pint.quantify("kg/kg").pint.to("g/kg").pint.dequantify()
        )
        θ_l = thermo.get_theta_l(
            ds_mtrgm.T, ds_mtrgm.P, (ds_mtrgm.QC + ds_mtrgm.QR + ds_mtrgm.QV)
        )
        ds_mtrgm["θl"] = θ_l
        grad = (
            ds_mtrgm["θl"].isel(height_2=slice(0, -1)).values
            - ds_mtrgm["θl"].isel(height_2=slice(1, None)).values
        )

        grad_ = np.empty((grad.shape[0], grad.shape[1] + 1))
        grad_[:, :] = np.nan
        grad_[:, 1:] = grad

        ds_mtrgm["θl_gradient"] = xr.DataArray(
            grad_,
            dims=("time", "height_2"),
            coords={"height_2": ds_mtrgm["θl"].height_2.values},
        )
        max_grad_idx = (
            ds_mtrgm["θl_gradient"].sel(height_2=slice(4, 0.8)).argmax(dim="height_2")
        )
        max_grad_height = (
            ds_mtrgm["θl_gradient"].sel(height_2=slice(4, 0.8)).height_2[max_grad_idx]
        )
        inv_height = max_grad_height.groupby(max_grad_height.time.dt.date).median()
        inv_heights[dom] = inv_height
        ds_mtrgm.close()

    dom = "obs"
    ds_rad = cat.radiosondes.bco.to_dask()
    ds_rad = ds_rad.swap_dims({"sounding": "launch_time"})
    ds_rad = ds_rad.drop(["sounding", "flight_time", "lat", "lon"])
    ds_mtrgm = ds_rad
    ds_mtrgm = ds_mtrgm.rename({"alt": "height"})
    ds_mtrgm["height"] = (
        (ds_mtrgm.height / 1000).pint.quantify() * pint.unit.Unit("km")
    ).pint.dequantify()
    ds_mtrgm["qt"] = ds_mtrgm.q.astype("float64")
    ds_mtrgm["qt"] = ds_mtrgm["qt"].pint.quantify("kg/kg").pint.dequantify()
    θ_l = thermo.get_theta_l(
        ds_mtrgm.ta.astype("float64"), ds_mtrgm.p.astype("float64"), ds_mtrgm["qt"]
    )
    ds_mtrgm["θl"] = θ_l.astype("float64")
    grad = (
        ds_mtrgm["θl"].isel(height=slice(0, -1)).values
        - ds_mtrgm["θl"].isel(height=slice(1, None)).values
    )

    grad_ = np.empty((grad.shape[0], grad.shape[1] + 1))
    grad_[:, :] = np.nan
    grad_[:, 1:] = grad

    ds_mtrgm["θl_gradient"] = xr.DataArray(
        grad_,
        dims=("launch_time", "height"),
        coords={"height": ds_mtrgm["θl"].height.values},
    )
    max_grad_idx = (
        ds_mtrgm["θl_gradient"]
        .sel(height=slice(0.8, 4))
        .dropna(how="all", dim="launch_time")
        .argmax(dim="height")
    )
    max_grad_height = (
        ds_mtrgm["θl_gradient"]
        .sel(height=slice(0.8, 4))
        .dropna(how="all", dim="launch_time")
        .height[max_grad_idx]
    )
    inv_height = max_grad_height.groupby(max_grad_height.launch_time.dt.date).median()
    inv_heights[dom] = inv_height
    ds_mtrgm.close()
    # -

    # Assuming observations are available, select only those that have model output
    ds_sel = ds_out.sel(time=~ds_out.control_CF_ICON_DOM02.isnull().any(dim="height"))

    # +
    patterns_to_show = ["Sugar", "Gravel", "Flowers", "Fish"]
    include_error = True
    bootstrap_samples = 10
    total_extra = True

    if total_extra is False:
        n_subplots = len(patterns_to_show) + 1
    else:
        n_subplots = len(patterns_to_show)

    for setup in ["control"]:  # ['highCCN', 'control', 'both']:
        fig, axs = plt.subplots(
            1, n_subplots, sharey=False, sharex=False, figsize=(10, 2), dpi=300
        )
        for p, pattern in enumerate(patterns_to_show):
            if total_extra is False:
                p += 1
            grp = ds_sel.where(ds_sel.pattern == pattern, drop=True)

            mean = grp.mean(dim="time")
            std = grp.std(dim="time")
            axs[p].set_title(pattern)
            (l1,) = axs[p].plot(mean.CF_kaband, mean.range / 1000, color="k")
            if setup in ["highCCN", "both"]:
                (l2,) = axs[p].plot(
                    mean.highCCN_CF_ICON_DOM01,
                    mean.height / 1000,
                    color="r",
                    linestyle=":",
                )
                (l3,) = axs[p].plot(
                    mean.highCCN_CF_ICON_DOM02,
                    mean.height / 1000,
                    color="r",
                    linestyle="-",
                )
            if setup in ["control", "both"]:
                (l2,) = axs[p].plot(
                    mean.control_CF_ICON_DOM01,
                    mean.height / 1000,
                    color=conf_dict["DOM01"]["color"],
                    linestyle="-",
                )
                (l3,) = axs[p].plot(
                    mean.control_CF_ICON_DOM02,
                    mean.height / 1000,
                    color=conf_dict["DOM02"]["color"],
                    linestyle="-",
                )
            if include_error:
                axs[p].fill_betweenx(
                    mean.range / 1000,
                    mean.CF_kaband
                    - sem(resample(grp.CF_kaband, n_samples=bootstrap_samples), axis=0),
                    mean.CF_kaband
                    + sem(resample(grp.CF_kaband, n_samples=bootstrap_samples), axis=0),
                    color="gray",
                    alpha=0.6,
                    edgecolor=None,
                )
                if setup in ["highCCN", "both"]:
                    axs[p].fill_betweenx(
                        mean.height / 1000,
                        mean.highCCN_CF_ICON_DOM01
                        - sem(grp.highCCN_CF_ICON_DOM01, axis=0),
                        mean.highCCN_CF_ICON_DOM01
                        + sem(grp.highCCN_CF_ICON_DOM01, axis=0),
                        color="gray",
                        alpha=0.6,
                    )
                    axs[p].fill_betweenx(
                        mean.height / 1000,
                        mean.highCCN_CF_ICON_DOM02
                        - sem(grp.highCCN_CF_ICON_DOM02, axis=0),
                        mean.highCCN_CF_ICON_DOM02
                        + sem(grp.highCCN_CF_ICON_DOM02, axis=0),
                        color="orange",
                        alpha=0.6,
                    )
                if setup in ["control", "both"]:
                    axs[p].fill_betweenx(
                        mean.height / 1000,
                        mean.control_CF_ICON_DOM01
                        - sem(
                            resample(
                                grp.control_CF_ICON_DOM01, n_samples=bootstrap_samples
                            ),
                            axis=0,
                        ),
                        mean.control_CF_ICON_DOM01
                        + sem(
                            resample(
                                grp.control_CF_ICON_DOM01, n_samples=bootstrap_samples
                            ),
                            axis=0,
                        ),
                        color=conf_dict["DOM01"]["color"],
                        alpha=0.6,
                        edgecolor=None,
                    )
                    axs[p].fill_betweenx(
                        mean.height / 1000,
                        mean.control_CF_ICON_DOM02
                        - sem(
                            resample(
                                grp.control_CF_ICON_DOM02, n_samples=bootstrap_samples
                            ),
                            axis=0,
                        ),
                        mean.control_CF_ICON_DOM02
                        + sem(
                            resample(
                                grp.control_CF_ICON_DOM02, n_samples=bootstrap_samples
                            ),
                            axis=0,
                        ),
                        color=conf_dict["DOM02"]["color"],
                        alpha=0.6,
                        edgecolor=None,
                    )
            axs[p].set_ylim(0, 5)
            axs[p].set_xlim(0, 55)
            axs[p].annotate(f"$N={len(grp.time)}$", (40, 0), fontsize=7)
            if p == 0:
                axs[p].set_ylabel("height / km")
            if p == 2:
                axs[p].set_xlabel("echo fraction / %")

            axs[p].set_xticks([np.round(mean.CF_kaband.max(), 1)])
            axs[p].set_xticks(np.arange(0, 51, 10), minor=True)
            axs[p].set_yticks(np.arange(0, 6, 1), minor=True)
            common_dates = list(
                set(grp.time.dt.date.values).intersection(inv_heights["obs"].date.values)
            )
            axs[p].set_yticks(
                [
                    np.round(
                        mean.CF_kaband.isel(range=mean.CF_kaband.argmax()).range.values
                        / 1000,
                        1,
                    ),
                    np.round(
                        inv_heights["obs"].sel(date=common_dates).median().item(0), 1
                    ),
                ]
            )

            add_ax = ph.add_twin(
                axs[p],
                color=conf_dict["DOM01"]["color"],
                labelcolor=conf_dict["DOM01"]["color"],
                width=2,
                direction="in",
                size=6,
            )
            add_ax.set_yticks(
                [
                    np.round(
                        mean.control_CF_ICON_DOM01.isel(
                            height=mean.control_CF_ICON_DOM01.argmax()
                        ).height.values
                        / 1000,
                        3,
                    ),
                    inv_heights[1].sel(date=grp.time).median().item(0),
                ],
                [],
            )
            add_ax.set_ylim(0, 5)

            add_ax = ph.add_twin(
                axs[p],
                color=conf_dict["DOM02"]["color"],
                labelcolor=conf_dict["DOM02"]["color"],
                width=2,
                direction="in",
                size=4,
            )
            add_ax.set_yticks(
                [
                    np.round(
                        mean.control_CF_ICON_DOM02.isel(
                            height=mean.control_CF_ICON_DOM02.argmax()
                        ).height.values
                        / 1000,
                        3,
                    ),
                    inv_heights[2].sel(date=grp.time).median().item(0),
                ],
                [],
            )
            add_ax.set_ylim(0, 5)
            add_ax = ph.add_twin_bottom(
                axs[p],
                color=conf_dict["DOM01"]["color"],
                labelcolor=conf_dict["DOM01"]["color"],
                width=2,
                direction="in",
            )
            add_ax.set_xticks([np.round(mean.control_CF_ICON_DOM01.max(), 3)], [])
            add_ax.set_xlim(0, 55)

            add_ax = ph.add_twin_bottom(
                axs[p],
                color=conf_dict["DOM02"]["color"],
                labelcolor=conf_dict["DOM02"]["color"],
                width=2,
                direction="in",
            )
            add_ax.set_xticks([np.round(mean.control_CF_ICON_DOM02.max(), 3)], [])
            add_ax.set_xlim(0, 55)

            # add_ax = ph.add_twin_bottom(axs[p], color=conf_dict['DOM03']['color'],
            #                 labelcolor=conf_dict['DOM03']['color'], width=2, direction='in')
            # add_ax.set_xticks([np.round(mean.control_CF_ICON_DOM03.max(),3)], [])
            # add_ax.set_xlim(0,55)

        #     plt.show()
        if total_extra is False:
            p = 0
            mean = ds_sel.mean(dim="time")
            grp = ds_sel
            std = ds_sel.std(dim="time")
            axs[p].set_title("total")
            (l1,) = axs[p].plot(mean.CF_kaband, mean.range / 1000, color="k")
            if setup in ["highCCN", "both"]:
                (l2,) = axs[p].plot(
                    mean.highCCN_CF_ICON_DOM01,
                    mean.height / 1000,
                    color="r",
                    linestyle=":",
                )
                (l3,) = axs[p].plot(
                    mean.highCCN_CF_ICON_DOM02,
                    mean.height / 1000,
                    color="r",
                    linestyle="-",
                )
            if setup in ["control", "both"]:
                (l2,) = axs[p].plot(
                    mean.control_CF_ICON_DOM01,
                    mean.height / 1000,
                    color=conf_dict["DOM01"]["color"],
                    linestyle="-",
                )
                (l3,) = axs[p].plot(
                    mean.control_CF_ICON_DOM02,
                    mean.height / 1000,
                    color=conf_dict["DOM02"]["color"],
                    linestyle="-",
                )

            if include_error:
                axs[p].fill_betweenx(
                    mean.range / 1000,
                    np.max(
                        [
                            np.zeros(len(mean.CF_kaband)),
                            mean.CF_kaband
                            - sem(
                                resample(ds_sel.CF_kaband, n_samples=bootstrap_samples),
                                axis=0,
                            ),
                        ],
                        axis=0,
                    ),
                    mean.CF_kaband
                    + sem(
                        resample(ds_sel.CF_kaband, n_samples=bootstrap_samples), axis=0
                    ),
                    color="gray",
                    alpha=0.4,
                    edgecolor=None,
                )
                if setup in ["highCCN", "both"]:
                    axs[p].fill_betweenx(
                        mean.height / 1000,
                        np.max(
                            [
                                np.zeros(len(mean.highCCN_CF_ICON_DOM01)),
                                mean.highCCN_CF_ICON_DOM01
                                - sem(
                                    resample(
                                        ds_sel.highCCN_CF_ICON_DOM01,
                                        n_samples=bootstrap_samples,
                                    ),
                                    axis=0,
                                ),
                            ],
                            axis=0,
                        ),
                        mean.highCCN_CF_ICON_DOM01
                        + sem(
                            resample(
                                ds_sel.highCCN_CF_ICON_DOM01, n_samples=bootstrap_samples
                            ),
                            axis=0,
                        ),
                        color="gray",
                        alpha=0.6,
                    )
                    axs[p].fill_betweenx(
                        mean.height / 1000,
                        np.max(
                            [
                                np.zeros(len(mean.highCCN_CF_ICON_DOM02)),
                                mean.highCCN_CF_ICON_DOM02
                                - sem(
                                    resample(
                                        ds_sel.highCCN_CF_ICON_DOM02,
                                        n_samples=bootstrap_samples,
                                    ),
                                    axis=0,
                                ),
                            ],
                            axis=0,
                        ),
                        mean.highCCN_CF_ICON_DOM02
                        + sem(
                            resample(
                                ds_sel.highCCN_CF_ICON_DOM02, n_samples=bootstrap_samples
                            ),
                            axis=0,
                        ),
                        color=conf_dict["DOM02"]["color"],
                        alpha=0.6,
                    )
                if setup in ["control", "both"]:
                    axs[p].fill_betweenx(
                        mean.height / 1000,
                        np.max(
                            [
                                np.zeros(len(mean.control_CF_ICON_DOM01)),
                                mean.control_CF_ICON_DOM01
                                - sem(
                                    resample(
                                        ds_sel.control_CF_ICON_DOM01,
                                        n_samples=bootstrap_samples,
                                    ),
                                    axis=0,
                                ),
                            ],
                            axis=0,
                        ),
                        mean.control_CF_ICON_DOM01
                        + sem(
                            resample(
                                ds_sel.control_CF_ICON_DOM01, n_samples=bootstrap_samples
                            ),
                            axis=0,
                        ),
                        color=conf_dict["DOM01"]["color"],
                        alpha=0.6,
                        edgecolor=None,
                    )
                    axs[p].fill_betweenx(
                        mean.height / 1000,
                        np.max(
                            [
                                np.zeros(len(mean.control_CF_ICON_DOM02)),
                                mean.control_CF_ICON_DOM02
                                - sem(
                                    resample(
                                        ds_sel.control_CF_ICON_DOM02,
                                        n_samples=bootstrap_samples,
                                    ),
                                    axis=0,
                                ),
                            ],
                            axis=0,
                        ),
                        mean.control_CF_ICON_DOM02
                        + sem(
                            resample(
                                ds_sel.control_CF_ICON_DOM02, n_samples=bootstrap_samples
                            ),
                            axis=0,
                        ),
                        color=conf_dict["DOM02"]["color"],
                        alpha=0.6,
                        edgecolor=None,
                    )

            add_ax = ph.add_twin(
                axs[p],
                color=conf_dict["DOM01"]["color"],
                labelcolor=conf_dict["DOM01"]["color"],
                width=2,
                direction="in",
                size=6,
            )
            add_ax.set_yticks(
                [
                    np.round(
                        mean.control_CF_ICON_DOM01.isel(
                            height=mean.control_CF_ICON_DOM01.argmax()
                        ).height.values
                        / 1000,
                        3,
                    ),
                    inv_heights[1].sel(date=grp.time).median().item(0),
                ],
                [],
            )
            add_ax.set_ylim(0, 5)

            add_ax = ph.add_twin(
                axs[p],
                color=conf_dict["DOM02"]["color"],
                labelcolor=conf_dict["DOM02"]["color"],
                width=2,
                direction="in",
                size=4,
            )
            add_ax.set_yticks(
                [
                    np.round(
                        mean.control_CF_ICON_DOM02.isel(
                            height=mean.control_CF_ICON_DOM02.argmax()
                        ).height.values
                        / 1000,
                        3,
                    ),
                    inv_heights[2].sel(date=grp.time).median().item(0),
                ],
                [],
            )
            add_ax.set_ylim(0, 5)

            try:
                add_ax = ph.add_twin(
                    axs[p],
                    color=conf_dict["DOM03"]["color"],
                    labelcolor=conf_dict["DOM03"]["color"],
                    width=2,
                    direction="in",
                    size=2,
                )
                common_dates = list(
                    set(grp.time.dt.date.values).intersection(inv_heights[3].date.values)
                )
                add_ax.set_yticks(
                    [
                        np.round(
                            mean.control_CF_ICON_DOM03.isel(
                                height=mean.control_CF_ICON_DOM03.argmax()
                            ).height.values
                            / 1000,
                            3,
                        ),
                        inv_heights[3].sel(date=common_dates).median().item(0),
                    ],
                    [],
                )
                add_ax.set_ylim(0, 5)
            except (ValueError, KeyError):
                add_ax.set_yticks([])

            add_ax = ph.add_twin_bottom(
                axs[p],
                color=conf_dict["DOM01"]["color"],
                labelcolor=conf_dict["DOM01"]["color"],
                width=2,
                direction="in",
            )
            add_ax.set_xticks([np.round(mean.control_CF_ICON_DOM01.max(), 3)], [])
            add_ax.set_xlim(0, 55)

            add_ax = ph.add_twin_bottom(
                axs[p],
                color=conf_dict["DOM02"]["color"],
                labelcolor=conf_dict["DOM02"]["color"],
                width=2,
                direction="in",
            )
            add_ax.set_xticks([np.round(mean.control_CF_ICON_DOM02.max(), 3)], [])
            add_ax.set_xlim(0, 55)

            axs[p].set_ylim(0, 5)
            axs[p].set_xlim(0, 55)
            axs[p].annotate(f"$N={len(ds_sel.time)}$", (40, 0), fontsize=7)

            axs[p].set_xticks([np.round(mean.CF_kaband.max(), 1)])
            axs[p].set_xticks(np.arange(0, 51, 10), minor=True)
            axs[p].set_yticks(np.arange(0, 6, 1), minor=True)
            common_dates = list(
                set(grp.time.dt.date.values).intersection(inv_heights["obs"].date.values)
            )
            axs[p].set_yticks(
                [
                    np.round(
                        mean.CF_kaband.isel(range=mean.CF_kaband.argmax()).range.values
                        / 1000,
                        1,
                    ),
                    np.round(
                        inv_heights["obs"].sel(date=common_dates).median().item(0), 1
                    ),
                ]
            )

        sns.despine(offset=2)
        fig.subplots_adjust(bottom=0.3, wspace=0.33)

        axs[2].legend(
            handles=[l1, l2, l3],
            labels=["cloud radar", "ICON 624m", "ICON 312m"],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.4),
            fancybox=False,
            shadow=False,
            ncol=4,
        )
        out_fn = (
            "FIG_CF_profile_{res}-variability_groupedbyPattern_"
            "{setup}_err-{err}_withoutTotal.pdf".format(
                res=resampling, setup=setup, err=include_error
            )
        )
        out_path = os.path.join(fig_output_folder, out_fn)
        out_dir = os.path.dirname(out_path)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        plt.savefig(out_path, bbox_inches="tight")
