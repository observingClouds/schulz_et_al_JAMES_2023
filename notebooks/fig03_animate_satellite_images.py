#!/work/mh0010/m300408/envs/covariability/bin/python
# SBATCH --partition=shared
# SBATCH --account=mh0010
# SBATCH --mem=6GB
# SBATCH --nodes=1
# SBATCH --time=1-12:30:00
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all
#     formats: ipynb,py:percent
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.2
#   kernelspec:
#     display_name: covariability
#     language: python
#     name: covariability
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.9.10
# ---

# %%
import gc
import os

import intake
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
import xarray as xr
from omegaconf import OmegaConf

cfg = OmegaConf.load("../config/paths.cfg")

sat_img_fmt = cfg.OBS.SATELLITES.GOES16.CH13.filename_fmt_glob
fig_output_fmt = (
    "../figures/fig04_animation/GOES16_comparison_SYN_ABI_CH13_%Y%m%d_%H%M.png"
)
cat_url = (
    "https://raw.githubusercontent.com/observingClouds/"
    "eurec4a-intake/ICON-LES-control-DOM03/catalog.yml"
)

synthetic_var = "synsat_rttov_forward_model_{DOM}__abi_ir__goes_16__channel_7"
goes16_var = "C13"
vmin = 270
vmax = 300
lats = slice(17, 7.5)
lons = slice(-60.25, -45)

cat = intake.open_catalog(cat_url)

outdir = os.path.dirname(fig_output_fmt)
if not os.path.isdir(outdir):
    os.makedirs(outdir)

# %%
syndat_ds_dict = {}
global indices
for DOM in tqdm.tqdm([1, 2, 3]):
    indices = {}
    if DOM in [1, 2]:
        syndat_ds_dict[DOM] = cat.simulations.ICON.LES_CampaignDomain_control[
            f"rttov_DOM0{DOM}"
        ].to_dask()
    else:
        syndat_ds_dict[DOM] = (
            cat.simulations.ICON.LES_CampaignDomain_control[f"rttov_DOM0{DOM}"]
            .to_dask()
            .sortby("time")
        )
        syndat_ds_dict[DOM]["time"] = syndat_ds_dict[DOM].time.dt.round("1s")

# %%
syndat_ds_dict[3].time

# %%
# Actual data
ds_sat_abi = xr.open_mfdataset(
    sat_img_fmt.format(date="*", time="*"),
    concat_dim="time",
    combine="nested",
    chunks={"time": 1},
    parallel=True,
)

# %%
for time in tqdm.tqdm(syndat_ds_dict[1].time):
    try:
        time_dt = pd.to_datetime(time.values)
    except BaseException:
        print("Error with time")
        print(time.values)
        continue

    if os.path.isfile(time_dt.strftime(fig_output_fmt)):
        continue

    fig = plt.figure(figsize=(8, 1.9), dpi=250)
    widths = [2, 2, 2, 2]
    heights = [3, 0.3]
    spec5 = fig.add_gridspec(
        ncols=4, nrows=2, width_ratios=widths, height_ratios=heights
    )
    axs = [None, None, None, None]
    axs[0] = ax = fig.add_subplot(spec5[0, 0])
    axs[1] = ax = fig.add_subplot(spec5[0, 1], sharey=axs[0])
    axs[2] = ax = fig.add_subplot(spec5[0, 2], sharey=axs[0])
    axs[3] = ax = fig.add_subplot(spec5[0, 3], sharey=axs[0])
    ax_cbar = ax = fig.add_subplot(spec5[1, :])
    plt.suptitle(time_dt.strftime("%Y-%m-%d %H:%M UTC"))
    p2 = (
        ds_sat_abi[goes16_var]
        .sel(time=time_dt, method="nearest")
        .sel(lat=lats, lon=lons)
        .plot(
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            ax=axs[0],
            cbar_ax=ax_cbar,
            cbar_kwargs={
                "extend": "both",
                "label": "brightness temperature / K",
                "orientation": "horizontal",
            },
        )
    )
    p1 = (
        syndat_ds_dict[1][synthetic_var.format(DOM=1)]
        .sel(time=time_dt)
        .plot(cmap="RdBu_r", vmin=vmin, vmax=vmax, ax=axs[1], add_colorbar=False)
    )
    try:
        p3 = (
            syndat_ds_dict[2][synthetic_var.format(DOM=2)]
            .sel(time=time_dt)
            .plot(cmap="RdBu_r", vmin=vmin, vmax=vmax, ax=axs[2], add_colorbar=False)
        )
        axs[2].set_title(None)
        axs[2].set_xlim(lons.start, lons.stop)
        axs[2].set_ylim(lats.start, lats.stop)
        axs[2].axes.set_aspect("equal")
        axs[2].set_ylabel(None)
    except KeyError:
        axs[2].axis("off")

    try:
        p4 = (
            syndat_ds_dict[3][synthetic_var.format(DOM=3)]
            .sel(time=time_dt)
            .plot(cmap="RdBu_r", vmin=vmin, vmax=vmax, ax=axs[3], add_colorbar=False)
        )
        axs[3].set_title(None)
        axs[3].set_xlim(lons.start, lons.stop)
        axs[3].set_ylim(lats.start, lats.stop)
        axs[3].axes.set_aspect("equal")
        axs[3].set_ylabel(None)
    except (KeyError, AttributeError):
        axs[3].axis("off")

    axs[0].set_title(None)
    axs[1].set_title(None)

    axs[0].set_ylabel(None)
    axs[1].set_ylabel(None)
    axs[2].set_ylabel(None)

    axs[0].set_xlabel(None)
    axs[1].set_xlabel(None)
    axs[2].set_xlabel(None)
    axs[3].set_xlabel(None)

    axs[0].set_xlim(lons.start, lons.stop)
    axs[0].set_ylim(lats.stop, lats.start)
    axs[1].set_xlim(lons.start, lons.stop)
    axs[1].set_ylim(lats.stop, lats.start)
    axs[2].set_xlim(lons.start, lons.stop)
    axs[2].set_ylim(lats.stop, lats.start)
    axs[3].set_xlim(lons.start, lons.stop)
    axs[3].set_ylim(lats.stop, lats.start)

    axs[0].axes.set_aspect("equal")
    axs[1].axes.set_aspect("equal")

    axs[1].tick_params(left=False, labelleft=False)
    axs[2].tick_params(left=False, labelleft=False)
    axs[3].tick_params(left=False, labelleft=False)

    plt.savefig(time_dt.strftime(fig_output_fmt), bbox_inches="tight")
    plt.close()
    gc.collect()

# %%
