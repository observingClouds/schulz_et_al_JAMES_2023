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

import datetime as dt
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr
from omegaconf import OmegaConf

sys.path.append("../src/helpers/")
import plot_helpers as ph  # noqa: E402

cfg = OmegaConf.load("../config/paths.cfg")
# -

ERA5_SST_fn = cfg.REANALYSIS.ERA5.sst.local
METEOR_SST_fn = cfg.OBS.Meteor.sst_dship

ds_ERA5 = xr.open_dataset(ERA5_SST_fn)
ds_METEOR = xr.open_dataset(METEOR_SST_fn)

print(ds_METEOR)

ERA5_MET = ds_ERA5.sel(
    latitude=ds_METEOR.lat,
    time=ds_METEOR.time,
    longitude=ds_METEOR.lon,
    method="nearest",
)

print(ERA5_MET)

median_ERA5 = ERA5_MET.median(dim="time")

# +
daily_mean = False
fig, axs = plt.subplots(1, 1, figsize=(8, 2), dpi=150)
plt.plot(ERA5_MET.time, ERA5_MET.sst, label="ERA5 SST", color="steelblue")
plt.plot(ERA5_MET.time, ERA5_MET.skt, label="ERA5 SKT", color="lightblue")
try:
    plt.plot(
        ds_METEOR.time,
        ds_METEOR.sea_surface_temperature + 273.15,
        label="R/V METEOR",
        color="k",
    )
except BaseException:
    if daily_mean:
        daily_mean_METEOR = ds_METEOR.groupby(ds_METEOR.time.dt.date).mean()
        data = np.nanmean(
            np.array(
                [
                    daily_mean_METEOR.SSTstar.values + 273.15,
                    daily_mean_METEOR.SSTport.values + 273.15,
                ]
            ),
            axis=0,
        )
        plt.plot(daily_mean_METEOR.date, data, label="R/V METEOR", color="k")
    else:
        data = np.nanmean(
            np.array(
                [ds_METEOR.SSTstar.values + 273.15, ds_METEOR.SSTport.values + 273.15]
            ),
            axis=0,
        )
        plt.plot(ds_METEOR.time, data, label="R/V METEOR", color="k")

date_fmt = mdates.DateFormatter("%m/%d")
axs.xaxis.set_major_formatter(date_fmt)
plt.legend(frameon=False, ncol=3)
plt.xlim(dt.datetime(2020, 1, 17), dt.datetime(2020, 2, 19))
plt.ylabel("temperature / K")
plt.xlabel("time / UTC mm/dd")
plt.ylim([26 + 273.15, 28.5 + 273.15])
plt.yticks([np.median(data)])
axs.tick_params(width=2, axis="y")
axs.set_yticks(
    np.arange(26, 29, 0.5) + 273, minor=True, labels=[299, "", "", "", "", 301.5]
)
axs.tick_params(axis="both", which="minor", labelsize=8)
ax_twin = ph.add_twin(
    axs, color="lightblue", labelcolor="lightblue", width=2, direction="out"
)
ax_twin.set_ylim([26 + 273.15, 28.5 + 273.15])
ax_twin.set_yticks([median_ERA5.skt])
ax_twin = ph.add_twin(
    axs, color="steelblue", labelcolor="steelblue", width=2, direction="out"
)
ax_twin.set_ylim([26 + 273.15, 28.5 + 273.15])
ax_twin.set_yticks([np.round(median_ERA5.sst, 2)])

sns.despine(offset=5)
plt.savefig(cfg.ANALYSIS.SST.OUTPUT.figure_filename, bbox_inches="tight")
