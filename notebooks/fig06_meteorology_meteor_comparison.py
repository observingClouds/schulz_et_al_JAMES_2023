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

import logging

import dask
import matplotlib
import matplotlib.pyplot as plt
import metpy.calc as mcalc
import metpy.units as units
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from intake import open_catalog
from omegaconf import OmegaConf
from scipy.stats import sem

cfg = OmegaConf.load("../config/paths.cfg")
# -

conf_dict = {
    "DOM01": {"label": "ICON 624m", "color": "#109AFA"},
    "DOM02": {"label": "ICON 312m", "color": "red"},
    "obs": {"label": "GOES-16 ABI", "color": "black"},
}

cat_address = (
    "https://raw.githubusercontent.com/eurec4a/eurec4a-intake/master/catalog.yml"
)
cat = open_catalog(cat_address)

meteor_navmet = xr.open_dataset(cfg.OBS.Meteor.sst_dship)

logging.info(meteor_navmet)

plt.plot(meteor_navmet.lon, meteor_navmet.lat)
plt.ylim(10, 15)
plt.xlim(-60, -55)

metgrm_east_DOM02 = (
    cat.simulations.ICON.LES_CampaignDomain_control.meteogram_c_east_DOM02.to_dask()
)
metgrm_east_DOM01 = (
    cat.simulations.ICON.LES_CampaignDomain_control.meteogram_c_east_DOM01.to_dask()
)

meteor_navmet_racetrack = meteor_navmet.sel(
    time=(
        (12.1 <= meteor_navmet.lat)
        & (meteor_navmet.lat <= 14.5)
        & (-57.5 <= meteor_navmet.lon)
        & (meteor_navmet.lon <= -57)
    )
)

common_dates = sorted(
    set(metgrm_east_DOM02.time.dt.date.values).intersection(
        meteor_navmet_racetrack.time.dt.date.values
    )
)

logging.info(f"Common dates {common_dates}")

meteor_navmet_racetrack = meteor_navmet_racetrack.sel(
    time=np.in1d(meteor_navmet_racetrack.time.dt.date, common_dates)
)

print(meteor_navmet_racetrack)

metgrm_east_DOM02 = metgrm_east_DOM02.sel(
    time=np.in1d(metgrm_east_DOM02.time.dt.date, common_dates)
)
metgrm_east_DOM01 = metgrm_east_DOM01.sel(
    time=np.in1d(metgrm_east_DOM01.time.dt.date, common_dates)
)

# +
temp_bins = np.arange(22, 30, 0.2) + 273.15
counts_meteor_port, bins = np.histogram(
    meteor_navmet_racetrack.Tport + 273.15, bins=temp_bins, density=True
)
counts_meteor_star, bins = np.histogram(
    meteor_navmet_racetrack.Tstar + 273.15, bins=temp_bins, density=True
)
counts_ICON312m, bins = np.histogram(metgrm_east_DOM02.T2M, bins=temp_bins, density=True)
counts_ICON624m, bins = np.histogram(metgrm_east_DOM01.T2M, bins=temp_bins, density=True)

fig, axs = plt.subplots(1, 1, figsize=(4, 3))
axs.stairs(
    counts_ICON312m * 100 / 5,
    bins,
    color=conf_dict["DOM02"]["color"],
    label="ICON-312m",
)
axs.stairs(
    counts_ICON624m * 100 / 5,
    bins,
    color=conf_dict["DOM01"]["color"],
    label="ICON-624m",
)
axs.stairs(
    counts_meteor_port * 100 / 5, bins, color=conf_dict["obs"]["color"], label="Meteor"
)
plt.legend()
plt.xlabel("temperature / K")
plt.ylabel("freq. of occurrence / %")
plt.tight_layout()
sns.despine()
plt.savefig("../figures/Meteor_vs_Simulation_temperature.pdf", bbox_inches="tight")
# -

logging.info(np.sum(counts_ICON312m * 100 / 5))

# +
counts_meteor_port, bins = np.histogram(
    meteor_navmet_racetrack.FF_true, bins=np.arange(0, 20), density=True
)
wsp = mcalc.wind_speed(
    metgrm_east_DOM02.U10M * units.units["m/s"],
    metgrm_east_DOM02.V10M * units.units["m/s"],
)
counts_ICON312m, bins = np.histogram(wsp, bins=np.arange(0, 20), density=True)
wsp = mcalc.wind_speed(
    metgrm_east_DOM01.U10M * units.units["m/s"],
    metgrm_east_DOM01.V10M * units.units["m/s"],
)
counts_ICON624m, bins = np.histogram(wsp, bins=np.arange(0, 20), density=True)

fig, axs = plt.subplots(1, 1, figsize=(4, 3))
axs.stairs(
    counts_ICON312m * 100, bins, color=conf_dict["DOM02"]["color"], label="ICON-312m"
)
axs.stairs(
    counts_ICON624m * 100, bins, color=conf_dict["DOM01"]["color"], label="ICON-624m"
)
axs.stairs(
    counts_meteor_port * 100, bins, color=conf_dict["obs"]["color"], label="Meteor"
)
# plt.legend()
plt.xlabel("wind speed / ms$^{-1}$")
plt.ylabel("freq. of occurrence / %")
plt.tight_layout()
sns.despine()
plt.savefig("../figures/Meteor_vs_Simulation_wspd.pdf", bbox_inches="tight")

# +
temp_bins = np.arange(25, 30, 0.2) + 273.15
counts_meteor_port, bins = np.histogram(
    meteor_navmet_racetrack.SSTport + 273.15, bins=temp_bins, density=True
)
counts_meteor_star, bins = np.histogram(
    meteor_navmet_racetrack.SSTstar + 273.15, bins=temp_bins, density=True
)
counts_ICON312m, bins = np.histogram(metgrm_east_DOM02.T_S, bins=temp_bins, density=True)
counts_ICON624m, bins = np.histogram(metgrm_east_DOM01.T_S, bins=temp_bins, density=True)

fig, axs = plt.subplots(1, 1, figsize=(4, 3))
axs.stairs(
    counts_ICON312m * 100 / 5,
    bins,
    color=conf_dict["DOM02"]["color"],
    label="ICON-312m (SKIN)",
)
axs.stairs(
    counts_ICON624m * 100 / 5,
    bins,
    color=conf_dict["DOM01"]["color"],
    label="ICON-624m (SKIN)",
)
axs.stairs(
    counts_meteor_port * 100 / 5,
    bins,
    color=conf_dict["obs"]["color"],
    label="Meteor (SST)",
)
plt.legend()
plt.xlabel("temperature / K")
plt.ylabel("freq. of occurrence / %")
plt.tight_layout()
sns.despine()
plt.savefig("../figures/Meteor_vs_Simulation_sst-skin.pdf", bbox_inches="tight")

# +
counts_meteor_port, bins = np.histogram(
    meteor_navmet_racetrack.RHport, bins=np.arange(50, 100), density=True
)
rh2m_dom01 = (
    mcalc.relative_humidity_from_dewpoint(
        metgrm_east_DOM01.T2M * units.units["K"],
        metgrm_east_DOM01.TD2M * units.units["K"],
    )
    * 100
)
rh2m_dom02 = (
    mcalc.relative_humidity_from_dewpoint(
        metgrm_east_DOM02.T2M * units.units["K"],
        metgrm_east_DOM02.TD2M * units.units["K"],
    )
    * 100
)
counts_ICON624m, bins = np.histogram(rh2m_dom01, bins=np.arange(50, 100), density=True)
counts_ICON312m, bins = np.histogram(rh2m_dom02, bins=np.arange(50, 100), density=True)

fig, axs = plt.subplots(1, 1, figsize=(4, 3))
axs.stairs(
    counts_ICON312m * 100, bins, color=conf_dict["DOM02"]["color"], label="ICON-312m"
)
axs.stairs(
    counts_ICON624m * 100, bins, color=conf_dict["DOM01"]["color"], label="ICON-624m"
)
axs.stairs(
    counts_meteor_port * 100, bins, color=conf_dict["obs"]["color"], label="Meteor"
)
# plt.legend()
plt.xlabel("relative humidity / %")
plt.ylabel("freq. of occurrence / %")
plt.tight_layout()
sns.despine()
plt.savefig("../figures/Meteor_vs_Simulation_rh.pdf", bbox_inches="tight")
