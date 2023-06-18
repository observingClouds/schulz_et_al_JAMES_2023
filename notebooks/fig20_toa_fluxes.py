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

import argparse
import logging

import bokeh
import dask
import holoviews as hv
import matplotlib as mpl
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

mpl.rcParams["text.usetex"] = True
mpl.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}"]  # for \text command


parser = argparse.ArgumentParser()
parser.add_argument(
    "--highClouds",
    action="store_true",
    help="Flag indicating whether high clouds should be included in output figure",
)
parser.add_argument(
    "--no-highClouds",
    dest="highClouds",
    action="store_false",
    help="Flag indicating whether high clouds should be included in output figure",
)
parser.set_defaults(highClouds=True)

args = parser.parse_args()

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

fn = "../data/intermediate/Quantile_brightnessT_GOES16.nc"
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

logging.info("SW-CRE mean")
df_cre_sw = df[["swCRE_daily_CERES", "swCRE_daily_DOM01", "swCRE_daily_DOM02"]]
logging.info(df_cre_sw.mean())

logging.info("LW-CRE mean")
df_cre_lw = df[["lwCRE_daily_CERES", "lwCRE_daily_DOM01", "lwCRE_daily_DOM02"]]
logging.info(df_cre_lw.mean())

logging.info("net-CRE mean")
df_cre_net = df[["netCRE_daily_CERES", "netCRE_daily_DOM01", "netCRE_daily_DOM02"]]
logging.info(df_cre_net.mean())

logging.info("Shallow cloud days only statistics")
logging.info(df_net.loc[DOM02_CERES_common_days_without_highclouds].corr())

logging.info(df_net.loc[DOM02_CERES_common_days_without_highclouds].mean())

logging.info("SW-CRE mean")
df_cre_sw = df[["swCRE_daily_CERES", "swCRE_daily_DOM01", "swCRE_daily_DOM02"]]
logging.info(df_cre_sw.loc[DOM02_CERES_common_days_without_highclouds].mean())

logging.info("LW-CRE mean")
df_cre_lw = df[["lwCRE_daily_CERES", "lwCRE_daily_DOM01", "lwCRE_daily_DOM02"]]
logging.info(df_cre_lw.loc[DOM02_CERES_common_days_without_highclouds].mean())

logging.info("net-CRE mean")
df_cre_net = df[["netCRE_daily_CERES", "netCRE_daily_DOM01", "netCRE_daily_DOM02"]]
logging.info(df_cre_net.loc[DOM02_CERES_common_days_without_highclouds].mean())

# ## Visualize cre

# ### Ignoring days with high clouds

# +
cre_min = 0
cre_max = 0

max_freq = ds_max["max_freq"].loc[DOM02_CERES_common_days_without_highclouds]
max_pattern = ds_max["max_pattern"].loc[DOM02_CERES_common_days_without_highclouds]
mean_pattern_freq = ds_max["mean_freq"].loc[DOM02_CERES_common_days_without_highclouds]
threshold_freq = threshold_freq_default

fig, axs_ = plt.subplots(2, 2, figsize=(8, 8))
axs = axs_.flatten()
for pr, (product, product_label) in enumerate(
    zip(["sw", "lw", "net"], ["shortwave", "longwave", "net"])  # noqa: B905
):
    p = pr + 1
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
    ceres_dom02_mean = np.mean(
        df[kf.format(product, "CERES")].loc[DOM02_CERES_common_days_without_highclouds]
    )
    ceres_dom02 = (
        df[kf.format(product, "CERES")].loc[DOM02_CERES_common_days_without_highclouds]
        - ceres_dom02_mean
    )
    dom02_mean = np.mean(
        df[kf.format(product, "DOM02")].loc[DOM02_CERES_common_days_without_highclouds]
    )
    dom02 = (
        df[kf.format(product, "DOM02")].loc[DOM02_CERES_common_days_without_highclouds]
        - dom02_mean
    )

    if args.highClouds:
        dom01_all = df[kf.format(product, "DOM01")] - np.mean(
            df[kf.format(product, "DOM01")]
        )
        ceres_dom01_all = df[kf.format(product, "CERES")] - np.mean(
            df[kf.format(product, "CERES")]
        )
        ceres_dom02_all = df[kf.format(product, "CERES")].loc[
            common_times_DOM02_CERES
        ] - np.mean(df[kf.format(product, "CERES")].loc[common_times_DOM02_CERES])
        dom02_all = df[kf.format(product, "DOM02")].loc[
            common_times_DOM02_CERES
        ] - np.mean(df[kf.format(product, "DOM02")].loc[common_times_DOM02_CERES])

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
    if args.highClouds:
        y_pos = -30
    else:
        y_pos = -28

    axs[p].text(
        -0,
        -22,
        r"$\overline{\mathrm{CRE}_{\mathrm{CERES}}}=$" + f"{ceres_dom02_mean:.2f}",
    )
    axs[p].text(
        -3.4,
        -25,
        r"$\overline{\mathrm{CRE}_{\mathrm{ICON}\textsf{-}\mathrm{312m}}}=$"
        + f"{dom02_mean:.2f}",
    )
    axs[p].text(
        -12,
        y_pos,
        r"$\mathrm{CRE}'_{\mathrm{CERES}}$"
        + f"={slope:.2f}"
        + r"$\cdot \mathrm{CRE'}_{\mathrm{ICON}\textsf{-}\mathrm{312m}}$",
    )

    # with high clouds
    if args.highClouds:
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

        slope, intercept, _, _, _ = scipy.stats.linregress(
            x=dom02_all, y=ceres_dom02_all
        )
        axs[p].plot(
            [np.min(dom02_all), np.max(dom02_all)],
            [regression_func(f) for f in (np.min(dom02_all), np.max(dom02_all))],
            color="black",
            alpha=0.2,
        )
        axs[p].text(
            -15,
            -33,
            r"CRE$'_{\mathrm{CERES}}$"
            + f"={slope:.2f}"
            + r"$\cdot \mathrm{CRE'}_{\mathrm{CERES}}$",
            alpha=0.2,
        )

    axs[p].set_aspect(1)

    min_cre = np.nanmin(np.hstack([dom01, dom02, ceres_dom01, ceres_dom02]))
    max_cre = np.nanmax(np.hstack([dom01, dom02, ceres_dom01, ceres_dom02]))
    if min_cre < cre_min:
        cre_min = min_cre - 1
    if max_cre > cre_max:
        cre_max = max_cre + 1
    cre_min = -30
    axs[p].set_ylabel(
        r"$\mathrm{CRE}_{\mathrm{CERES}}-\overline{\mathrm{CRE}_{\mathrm{CERES}}}$ /"
        r" Wm$^{-2}$"
    )
    axs[p].set_xlabel(
        r"$\mathrm{CRE}_{\mathrm{ICON-312m}}-\overline{\mathrm{CRE}_{\mathrm{ICON-312m}}}$"
        r" / Wm$^{-2}$"
    )

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
cre_max = 40

fig, axs_ = plt.subplots(2, 2, figsize=(8, 8))
axs = axs_.flatten()[0]

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
if args.highClouds:
    y_pos = -8
else:
    y_pos = -13
axs.text(
    5,
    y_pos,
    r"TOA$_{\mathrm{CERES}}$"
    + f"={slope:.2f}"
    + r"$\cdot \mathrm{TOA}_{\mathrm{ICON}\textsf{-}\mathrm{312m}}$"
    + f"+{intercept:.2f}",
)

# with high clouds
if args.highClouds:
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
                mean_pattern_freq.pattern.values[
                    max_pattern.sel(date=date.to_datetime64())
                ]
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
axs.set_ylabel("net TOA (CERES) / Wm$^{-2}$")
axs.set_xlabel("net TOA (ICON-312m) / Wm$^{-2}$")

tooltip = mpld3.plugins.PointLabelTooltip(
    scatter, labels=["{0}".format(i.strftime("%d.%m.%Y")) for i in sim.index]
)
mpld3.plugins.connect(fig, tooltip)

axs.set_ylim(cre_min, cre_max)
axs.set_xlim(cre_min, cre_max)
axs.set_yticks(np.arange(0, 50, 20))
axs.set_xticks(np.arange(0, 50, 20))
axs.plot([cre_max, cre_min], [cre_max, cre_min], color="lightgrey")

plt.tight_layout()
sns.despine()
mpld3.save_html(fig, "../figures/netTOA_scatter_obs_vs_dom02.html")
plt.savefig(
    "../figures/Daily_netTOA_obs_vs_sim_dom02.pdf",
    bbox_inches="tight",
)
