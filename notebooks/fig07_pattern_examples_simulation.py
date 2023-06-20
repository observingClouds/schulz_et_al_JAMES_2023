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

import datetime
import os

import matplotlib.pyplot as plt
import xarray as xr
from omegaconf import OmegaConf

cfg = OmegaConf.load("../config/paths.cfg")
sat_path = cfg.OBS.SATELLITES.GOES16.CH13.filename_fmt
output_folder = "../figures/fig09"

dic = {
    "Sugar": {"ABI": datetime.datetime(2020, 2, 6, 15, 0).strftime(sat_path)},
    "Gravel": {"ABI": datetime.datetime(2020, 1, 13, 18, 0).strftime(sat_path)},
    "Flowers": {"ABI": datetime.datetime(2020, 2, 12, 18, 0).strftime(sat_path)},
    "Fish": {"ABI": datetime.datetime(2020, 1, 22, 14, 0).strftime(sat_path)},
}

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for p, p_dic in dic.items():
    fig = plt.figure(dpi=150)
    fig.add_axes([0, 0, 1, 1])
    ds_ABI = xr.open_dataset(p_dic["ABI"])
    ds_ABI.C13.sel(lat=slice(17.2, 10), lon=slice(-60, -50.4)).plot(
        cmap="RdBu_r", vmin=275, vmax=300, add_colorbar=False
    )
    plt.title(None)
    plt.axis("off")
    plt.savefig(
        "../figures/fig09/GOES-16_ABI_pattern_sample_{}.png".format(p),
        bbox_inches="tight",
        dpi=150,
    )
    plt.show()
