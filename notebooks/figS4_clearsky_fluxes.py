import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from omegaconf import OmegaConf

cfg = OmegaConf.load("./config/paths.cfg")
params = OmegaConf.load("./config/mesoscale_params.yaml")

CRE_output_fmt = cfg.ANALYSIS.CRE.output_filename_fmt

ds_max = xr.open_dataset("./data/result/max_pattern_freq.nc")

color_dict = {
    "Sugar": "#A1D791",
    "Flower": "#93D2E2",
    "Fish": "#2281BB",
    "Gravel": "#3EAE47",
    "Flowers": "#93D2E2",
    "Unclassified": "grey",
}

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler("./logs/figS04.log"),
        logging.StreamHandler(),
    ],
)
# -

max_freq_all = ds_max["max_freq"]
max_pattern_all = ds_max["max_pattern"]
mean_pattern_freq_all = ds_max["mean_freq"]
threshold_freq_default = params.manual_classifications.threshold_pattern
threshold_freq_all = threshold_freq_default

kf = "{}CRE_daily_{}"
style_dict = {
    "CERES": {"marker": "x", "color": "black", "label": "CERES"},
    "DOM01": {"marker": "o", "color": "red", "label": "ICON-624m"},
    "DOM02": {"marker": ".", "color": "orange", "label": "ICON-312m"},
}

df = pd.read_json(CRE_output_fmt).loc["2020-01-11":"2020-02-18"]

df_nohighClouds = (
    pd.read_parquet("./data/result/no_high_clouds_DOM02.pq")
    .set_index("no_high_cloud")
    .loc["2020-01-11":"2020-02-18"]
    .reset_index()
)

colors = []
for date in df.loc[df_nohighClouds["no_high_cloud"].iloc[1:]].sort_index().index.values:
    if max_freq_all.sel(date=date) > threshold_freq_all:
        color = color_dict[
            mean_pattern_freq_all.pattern.values[max_pattern_all.sel(date=date)]
        ]
        colors.append(color)
    else:
        colors.append("grey")

fig, axs = plt.subplots(1, 2, figsize=(8, 8))
axs[0].scatter(
    df.loc[df_nohighClouds["no_high_cloud"].iloc[1:]].sort_index().swclear_daily_DOM02,
    df.loc[df_nohighClouds["no_high_cloud"].iloc[1:]].sort_index().swclear_daily_CERES,
    color=colors,
)
axs[0].set_aspect(1)
axs[0].set_ylim(290, 360)
axs[0].set_xlim(290, 360)
axs[0].plot([360, 290], [360, 290], "grey", linestyle="--", linewidth=1)
axs[0].set_xlabel("net\nclearsky SW(ICON) / Wm$^{-2}$")
axs[0].set_ylabel("net\nclearsky SW(CERES) / Wm$^{-2}$")
axs[0].set_yticks(np.arange(300, 361, 20))
axs[1].scatter(
    df.loc[df_nohighClouds["no_high_cloud"].iloc[1:]].sort_index().lwclear_daily_DOM02,
    df.loc[df_nohighClouds["no_high_cloud"].iloc[1:]].sort_index().lwclear_daily_CERES,
    color=colors,
)
axs[1].set_aspect(1)
axs[1].plot([-310, -285], [-310, -285], "grey", linestyle="--", linewidth=1)
axs[1].set_xlabel("net\nclearsky LW(ICON) / Wm$^{-2}$")
axs[1].set_ylabel("net\nclearsky LW(CERES) / Wm$^{-2}$")
axs[1].set_yticks(np.arange(-290, -311, -10))
axs[1].set_xticks(np.arange(-290, -311, -10))
sns.despine(offset=5)
plt.tight_layout()
plt.savefig("./figures/figureS04.pdf", bbox_inches="tight")
