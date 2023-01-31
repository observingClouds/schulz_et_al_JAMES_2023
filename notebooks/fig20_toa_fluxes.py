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

import bokeh
import dask
import holoviews as hv
import matplotlib.pyplot as plt
import mpld3
import numpy as np
import pandas as pd
import panel as pn
import scipy
import seaborn as sns
import xarray as xr
from bokeh.plotting import figure
from holoviews import opts
from holoviews.plotting.links import DataLink
from intake import open_catalog
from omegaconf import OmegaConf

hv.extension("bokeh")

cfg = OmegaConf.load("../config/paths.cfg")
params = OmegaConf.load("../config/mesoscale_params.yaml")

geobounds = {}
geobounds["lat_min"] = params.metrics.geobounds.lat_min
geobounds["lat_max"] = params.metrics.geobounds.lat_max
geobounds["lon_min"] = params.metrics.geobounds.lon_min
geobounds["lon_max"] = params.metrics.geobounds.lon_max

threshold_discard_percentile = params.metrics.BTbounds.threshold_discard_percentile
threshold_discard_temperature = params.metrics.BTbounds.threshold_discard_temperature

cat_tmp = open_catalog(
    "https://raw.githubusercontent.com/eurec4a/eurec4a-intake/master/catalog.yml"
)

CRE_output_fmt = cfg.ANALYSIS.CRE.output_filename_fmt

ds_max = xr.open_dataset("../data/result/max_pattern_freq.nc")

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
        logging.FileHandler("../logs/fig20+fig21.log"),
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

df

# ## Get mask for high clouds in OBS

ds_bt_dom01 = cat_tmp.simulations.ICON.LES_CampaignDomain_control.rttov_DOM01.to_dask()
ds_bt_dom02 = cat_tmp.simulations.ICON.LES_CampaignDomain_control.rttov_DOM02.to_dask()

BT_DOM01 = ds_bt_dom01.sel(
    lon=slice(geobounds["lon_min"], geobounds["lon_max"]),
    lat=slice(geobounds["lat_min"], geobounds["lat_max"]),
).synsat_rttov_forward_model_1__abi_ir__goes_16__channel_7
BT_DOM02 = ds_bt_dom02.sel(
    lon=slice(geobounds["lon_min"], geobounds["lon_max"]),
    lat=slice(geobounds["lat_min"], geobounds["lat_max"]),
).synsat_rttov_forward_model_2__abi_ir__goes_16__channel_7

q_DOM01 = BT_DOM01.quantile(
    threshold_discard_percentile / 100, dim=["lat", "lon"]
).compute()
q_DOM02 = (
    BT_DOM02.chunk({"lat": 534, "lon": 533, "time": 12})
    .quantile(threshold_discard_percentile / 100, dim=["lat", "lon"])
    .compute()
)

fn = (
    f"../data/intermediate/Quantile_{threshold_discard_percentile}_brightnessT_GOES16.nc"
)
q_ABI = xr.open_dataset(fn)
q_ABI = q_ABI.sel(time=slice("2020-01-11", "2020-02-18"))


# ## Calculate average and correlations
# +
def regression_func(x):
    return slope * x + intercept


count_highcloud_occurrance_DOM01 = (
    (q_DOM01 < threshold_discard_temperature).resample(time="1D").sum()
)
days_without_highclouds_DOM01 = count_highcloud_occurrance_DOM01.where(
    count_highcloud_occurrance_DOM01 == 0, drop=True
).time
print(count_highcloud_occurrance_DOM01)
count_highcloud_occurrance_DOM02 = (
    (q_DOM02 < threshold_discard_temperature).resample(time="1D").sum()
)
days_without_highclouds_DOM02 = count_highcloud_occurrance_DOM02.where(
    count_highcloud_occurrance_DOM02 == 0, drop=True
).time
count_highcloud_occurrance_CERES = (
    (q_ABI.C13 < threshold_discard_temperature).resample(time="1D").sum()
)
days_without_highclouds_CERES = count_highcloud_occurrance_CERES.where(
    count_highcloud_occurrance_CERES == 0, drop=True
).time

common_times_DOM02_CERES = list(
    set(count_highcloud_occurrance_CERES.time.values).intersection(
        count_highcloud_occurrance_DOM02.time.values
    )
)

DOM01_CERES_common_days_without_highclouds = list(
    set(days_without_highclouds_CERES.values).intersection(
        days_without_highclouds_DOM01.values
    )
)
DOM02_CERES_common_days_without_highclouds = list(
    set(days_without_highclouds_CERES.values).intersection(
        days_without_highclouds_DOM02.values
    )
)

df_net = df[["net_daily_CERES", "net_daily_DOM01", "net_daily_DOM02"]]

logging.info("All day statistics")
logging.info(df_net.corr())

logging.info(df_net.mean())

logging.info("Shallow cloud days only statistics")
logging.info(df_net.loc[DOM02_CERES_common_days_without_highclouds].corr())

logging.info(df_net.loc[DOM02_CERES_common_days_without_highclouds].mean())

# ## Visualize cre

# ### Ignoring days with high clouds

# +
cre_min = 0
cre_max = 0

max_freq = ds_max["max_freq"].loc[DOM02_CERES_common_days_without_highclouds]
max_pattern = ds_max["max_pattern"].loc[DOM02_CERES_common_days_without_highclouds]
mean_pattern_freq = ds_max["mean_freq"].loc[DOM02_CERES_common_days_without_highclouds]
threshold_freq = threshold_freq_default

fig, axs = plt.subplots(1, 3, figsize=(12, 5))
for p, (product, product_label) in enumerate(
    zip(["sw", "lw", "net"], ["shortwave", "longwave", "net"])  # noqa: B905
):
    axs[p].set_title(product_label)
    dom01 = df[kf.format(product, "DOM01")].loc[
        DOM01_CERES_common_days_without_highclouds
    ] - np.mean(
        df[kf.format(product, "DOM01")].loc[DOM01_CERES_common_days_without_highclouds]
    )
    ceres_dom01 = df[kf.format(product, "CERES")].loc[
        DOM01_CERES_common_days_without_highclouds
    ] - np.mean(
        df[kf.format(product, "CERES")].loc[DOM01_CERES_common_days_without_highclouds]
    )
    ceres_dom02 = df[kf.format(product, "CERES")].loc[
        DOM02_CERES_common_days_without_highclouds
    ] - np.mean(
        df[kf.format(product, "CERES")].loc[DOM02_CERES_common_days_without_highclouds]
    )
    dom02 = df[kf.format(product, "DOM02")].loc[
        DOM02_CERES_common_days_without_highclouds
    ] - np.mean(
        df[kf.format(product, "DOM02")].loc[DOM02_CERES_common_days_without_highclouds]
    )

    dom01_all = df[kf.format(product, "DOM01")] - np.mean(
        df[kf.format(product, "DOM01")]
    )
    ceres_dom01_all = df[kf.format(product, "CERES")] - np.mean(
        df[kf.format(product, "CERES")]
    )
    ceres_dom02_all = df[kf.format(product, "CERES")].loc[
        common_times_DOM02_CERES
    ] - np.mean(df[kf.format(product, "CERES")].loc[common_times_DOM02_CERES])
    dom02_all = df[kf.format(product, "DOM02")].loc[common_times_DOM02_CERES] - np.mean(
        df[kf.format(product, "DOM02")].loc[common_times_DOM02_CERES]
    )

    # without high clouds
    colors = []
    for date in dom02.index:
        if max_freq.sel(date=date.to_datetime64()) > threshold_freq:
            color = color_dict[
                mean_pattern_freq.pattern.values[
                    max_pattern.sel(date=date.to_datetime64())
                ]
            ]
            colors.append(color)
        else:
            colors.append("grey")
    scatter = axs[p].scatter(dom02, ceres_dom02, color=colors)

    slope, intercept, _, _, _ = scipy.stats.linregress(x=dom02, y=ceres_dom02)
    axs[p].plot(
        [np.min(dom02), np.max(dom02)],
        [regression_func(f) for f in (np.min(dom02), np.max(dom02))],
        color="black",
    )
    axs[p].text(
        -15,
        -30,
        r"CRE$_{\mathrm{OBS}}$"
        + f"={slope:.2f}"
        + r"$\cdot \mathrm{CRE}_{\mathrm{SIM}}$"
        + f"+{intercept:.2f}",
    )

    # with high clouds
    colors = []
    for date in dom02_all.index:
        if max_freq_all.sel(date=date.to_datetime64()) > threshold_freq_all:
            color = color_dict[
                mean_pattern_freq_all.pattern.values[
                    max_pattern_all.sel(date=date.to_datetime64())
                ]
            ]
            colors.append(color)
        else:
            colors.append("grey")
    scatter = axs[p].scatter(
        dom02_all, ceres_dom02_all, color=colors, alpha=0.2, linewidth=0
    )

    slope, intercept, _, _, _ = scipy.stats.linregress(x=dom02_all, y=ceres_dom02_all)
    axs[p].plot(
        [np.min(dom02_all), np.max(dom02_all)],
        [regression_func(f) for f in (np.min(dom02_all), np.max(dom02_all))],
        color="black",
        alpha=0.2,
    )
    axs[p].text(
        -15,
        -33,
        r"CRE$_{\mathrm{OBS}}$"
        + f"={slope:.2f}"
        + r"$\cdot \mathrm{CRE}_{\mathrm{SIM}}$"
        + f"+{intercept:.2f}",
        alpha=0.2,
    )

    axs[p].set_aspect(1)

    min_cre = np.nanmin(np.hstack([dom01, dom02, ceres_dom01, ceres_dom02]))
    max_cre = np.nanmax(np.hstack([dom01, dom02, ceres_dom01, ceres_dom02]))
    if min_cre < cre_min:
        cre_min = min_cre - 1
    if max_cre > cre_max:
        cre_max = max_cre + 1
    if p == 0:
        axs[p].set_ylabel("daily CRE anomaly of observations / Wm-2")
    if p == 1:
        axs[p].set_xlabel("daily CRE anomaly of simulations / Wm-2")

    tooltip = mpld3.plugins.PointLabelTooltip(
        scatter, labels=["{0}".format(i.strftime("%d.%m.%Y")) for i in dom02.index]
    )
    mpld3.plugins.connect(fig, tooltip)

for ax in axs:
    ax.set_ylim(cre_min, cre_max)
    ax.set_xlim(cre_min, cre_max)
    ax.plot([cre_max, cre_min], [cre_max, cre_min], color="lightgrey")


plt.tight_layout()
sns.despine()
mpld3.save_html(fig, "../figures/cre_scatter_obs_vs_dom02.html")
plt.savefig("../figures/Daily_CRE_anomaly_obs_vs_sim_dom02.pdf", bbox_inches="tight")
# -

# Figures shows that the simulations vary too much in CRE compared to observations

# +
DOM02_CERES_common_days = df_net.index

cre_min = -15
cre_max = 75

fig, axs = plt.subplots(1, 1)

# without high clouds
max_freq = ds_max["max_freq"].loc[DOM02_CERES_common_days_without_highclouds]
max_pattern = ds_max["max_pattern"].loc[DOM02_CERES_common_days_without_highclouds]
mean_pattern_freq = ds_max["mean_freq"].loc[DOM02_CERES_common_days_without_highclouds]
threshold_freq = threshold_freq_default

sim = df_net["net_daily_DOM02"].loc[DOM02_CERES_common_days_without_highclouds]
obs = df_net["net_daily_CERES"].loc[DOM02_CERES_common_days_without_highclouds]

colors = []
for date in sim.index:
    if max_freq.sel(date=date.to_datetime64()) > threshold_freq:
        color = color_dict[
            mean_pattern_freq.pattern.values[max_pattern.sel(date=date.to_datetime64())]
        ]
        colors.append(color)
    else:
        colors.append("grey")
scatter = axs.scatter(sim, obs, color=colors)

slope, intercept, _, _, _ = scipy.stats.linregress(x=sim, y=obs)
axs.plot(
    [np.min(sim), np.max(sim)],
    [regression_func(f) for f in (np.min(sim), np.max(sim))],
    color="black",
)
axs.text(
    20,
    -8,
    r"TOA$_{\mathrm{OBS}}$"
    + f"={slope:.2f}"
    + r"$\cdot \mathrm{TOA}_{\mathrm{SIM}}$"
    + f"+{intercept:.2f}",
)

# with high clouds
max_freq = ds_max["max_freq"].loc[DOM02_CERES_common_days]
max_pattern = ds_max["max_pattern"].loc[DOM02_CERES_common_days]
mean_pattern_freq = ds_max["mean_freq"].loc[DOM02_CERES_common_days]
threshold_freq = threshold_freq_default

sim = df_net["net_daily_DOM02"].loc[DOM02_CERES_common_days]
obs = df_net["net_daily_CERES"].loc[DOM02_CERES_common_days]

colors = []
for date in sim.index:
    if max_freq.sel(date=date.to_datetime64()) > threshold_freq:
        color = color_dict[
            mean_pattern_freq.pattern.values[max_pattern.sel(date=date.to_datetime64())]
        ]
        colors.append(color)
    else:
        colors.append("grey")
scatter = axs.scatter(sim, obs, color=colors, alpha=0.2)

slope, intercept, _, _, _ = scipy.stats.linregress(x=sim, y=obs)
axs.plot(
    [np.min(sim), np.max(sim)],
    [regression_func(f) for f in (np.min(sim), np.max(sim))],
    color="black",
    alpha=0.2,
)
axs.text(
    20,
    -13,
    r"TOA$_{\mathrm{OBS}}$"
    + f"={slope:.2f}"
    + r"$\cdot \mathrm{TOA}_{\mathrm{SIM}}$"
    + f"+{intercept:.2f}",
    alpha=0.2,
)

axs.set_aspect(1)
axs.set_ylabel("daily net TOA radiation of observations / Wm-2")
axs.set_xlabel("daily net TOA radiation of simulations / Wm-2")

tooltip = mpld3.plugins.PointLabelTooltip(
    scatter, labels=["{0}".format(i.strftime("%d.%m.%Y")) for i in sim.index]
)
mpld3.plugins.connect(fig, tooltip)

axs.set_ylim(cre_min, cre_max)
axs.set_xlim(cre_min, cre_max)
axs.set_yticks(np.arange(0, 70, 20))
axs.set_xticks(np.arange(0, 70, 20))
axs.plot([cre_max, cre_min], [cre_max, cre_min], color="lightgrey")

plt.tight_layout()
sns.despine()
mpld3.save_html(fig, "../figures/netTOA_scatter_obs_vs_dom02.html")
plt.savefig(
    "../figures/Daily_netTOA_obs_vs_sim_dom02.pdf",
    bbox_inches="tight",
)
