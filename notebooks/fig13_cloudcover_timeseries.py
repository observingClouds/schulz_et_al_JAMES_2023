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

import datetime
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

sys.path.append("../src/deps/")
sys.path.append("../src/helpers/")

import cluster_helpers as ch  # noqa: E402
import plot_helpers as ph  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.FileHandler("../logs/fig13.log"), logging.StreamHandler()],
)

cfg = OmegaConf.load("../config/paths.cfg")
params = OmegaConf.load("../config/mesoscale_params.yaml")
cat = open_catalog(
    "https://raw.githubusercontent.com/observingClouds/eurec4a-intake/simulations/catalog.yml"
)

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

if __name__ == "__main__":
    client = ch.setup_cluster("local cluster", verbose=logging.ERROR)
    print(client)

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

alphas = {"exp1": 0.5, "exp2": 1}
descr = {1: "high CCN", 2: "control"}

df_simulation = xr.open_dataset(
    cfg.ANALYSIS.MESOSCALE.METRICS.output_filename_fmt.format(
        type="rttov", DOM=1, exp=2
    )
)
df_simulation = df_simulation.isel(index=df_simulation.percentile_BT > 100)
times_sim = set(df_simulation.time.values)
times_obs = set(df_goes16_dom1.time.values)
common_times = sorted(times_obs.intersection(times_sim))
data_obs = df_goes16_dom1.set_index(index="time").sel(
    index=slice("2020-01-10", np.max(list(common_times)))
)

obs_1D_mean = data_obs.cloud_fraction.resample(index="1D").mean().compute()

df_simulation = xr.open_dataset(
    cfg.ANALYSIS.MESOSCALE.METRICS.output_filename_fmt.format(
        type="rttov", DOM=2, exp=2
    )
)
df_simulation = df_simulation.set_index(index="time")
DOM02_1D_mean = df_simulation.cloud_fraction.resample(index="1D").mean().compute()

times = list(set(obs_1D_mean.index.values).intersection(DOM02_1D_mean.index.values))

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

# + tags=[]
fig, axs = plt.subplots(1, 1, figsize=(8, 2.2), dpi=300)
df_simulation = xr.open_dataset(
    cfg.ANALYSIS.MESOSCALE.METRICS.output_filename_fmt.format(
        type="rttov", DOM=2, exp=2
    )
)
df_simulation = df_simulation.isel(index=df_simulation.percentile_BT > 100)
times_sim = set(df_simulation.time.values)
times_obs = set(df_goes16_dom1.time.values)
common_times = sorted(times_obs.intersection(times_sim))
data_obs = df_goes16_dom1.set_index(index="time").sel(
    index=slice("2020-01-12", np.max(list(common_times)))
)
axs.plot(
    data_obs.index,
    data_obs.cloud_fraction,
    label=conf_dict["obs"]["label"],
    color=conf_dict["obs"]["color"],
)

high_cloud_idx = np.where(data_obs.percentile_BT <= 290, 0.4, np.nan)
low_cloud_mask = np.where(data_obs.percentile_BT <= 290, False, True)
axs.plot(data_obs.index, high_cloud_idx, color="grey", alpha=0.2, linewidth=10)

obs_overall_mean = float(data_obs.cloud_fraction.median())
obs_overall_25th = float(data_obs.cloud_fraction.quantile(0.25))
obs_overall_75th = float(data_obs.cloud_fraction.quantile(0.75))

obs_lowcloud_mean = float(data_obs.sel(index=low_cloud_mask).cloud_fraction.median())
obs_lowcloud_25th = float(
    data_obs.sel(index=low_cloud_mask).cloud_fraction.quantile(0.25)
)
obs_lowcloud_75th = float(
    data_obs.sel(index=low_cloud_mask).cloud_fraction.quantile(0.75)
)


if __name__ == "__main__":
    logging.info(
        f"OBS: mean: {obs_overall_mean:.3f}, mean (without high clouds): {obs_lowcloud_mean:.3f}"
    )
times_obs = set(
    df_goes16_dom1.set_index(index="time")
    .sel(index=slice(np.min(list(common_times)), np.max(list(common_times))))
    .index.values
)
experiments = []
means = {}
for experiment in [2]:
    for domain in [1, 2]:
        experiments.append(f"exp{experiment}.dom{domain}")
        confs = conf_dict["DOM{DOM:02g}".format(DOM=domain)]
        try:
            df_simulation = xr.open_dataset(
                cfg.ANALYSIS.MESOSCALE.METRICS.output_filename_fmt.format(
                    type="rttov", DOM=domain, exp=experiment
                )
            )
        except FileNotFoundError:
            continue
        df_simulation = df_simulation.isel(index=df_simulation.percentile_BT > 100)
        times_sim = set(df_simulation.time.values)

        common_times = sorted(times_obs.intersection(times_sim))
        logging.debug("Resampling")
        data = (
            df_simulation.set_index(index="time")
            .cloud_fraction.drop_duplicates("index")
            .sortby("index")
            .sel(index=sorted(common_times))
            .resample(index="10T")
            .nearest(tolerance="10T")
        )
        logging.debug("Plotting")
        axs.plot(
            data.index,
            data,
            label=confs["label"],
            color=confs["color"],
            alpha=alphas["exp" + str(experiment)],
        )

        data_reindex = data.reindex(
            index=data_obs.index, method="nearest", tolerance="5T"
        )
        #     print(data_reindex.sel(index=low_cloud_mask).mean())
        means[f"DOM0{domain}_overall_mean"] = float(data.median())
        means[f"DOM0{domain}_overall_25th"] = float(data.quantile(0.25))
        means[f"DOM0{domain}_overall_75th"] = float(data.quantile(0.75))
        means[f"DOM0{domain}_lowclouds_mean"] = float(
            data_reindex.sel(index=low_cloud_mask).median()
        )
        means[f"DOM0{domain}_lowclouds_25th"] = float(
            data_reindex.sel(index=low_cloud_mask).quantile(0.25)
        )
        means[f"DOM0{domain}_lowclouds_75th"] = float(
            data_reindex.sel(index=low_cloud_mask).quantile(0.75)
        )
        if __name__ == "__main__":
            logging.info(
                f"{descr[experiment]} "
                f'DOM0{domain}: mean: {means[f"DOM0{domain}_overall_mean"]:.3f},'
                f'mean (without high clouds): {means[f"DOM0{domain}_lowclouds_mean"]:.3f}'
            )

plt.ylabel(r"$C_{\mathrm{B}}$")
plt.xlabel("date / UTC mm/dd")
plt.xlim(datetime.datetime(2020, 1, 12), datetime.datetime(2020, 2, 20))
axs.xaxis.set_major_formatter(md.DateFormatter(fmt="%m/%d"))
axs.set_ylim(0, 0.62)

ax = axs
ax.set_yticks([np.round(obs_overall_25th, 2), np.round(obs_overall_75th, 2)])
ax.set_yticks(np.arange(0, 0.62, 0.1), minor=True)
ax_obs_all_mean = ph.add_twin(ax, color="k", labelcolor="k", width=2, direction="out")
ax_obs_all_mean.set_yticks([np.round(obs_overall_mean, 2)])
ax_obs_all_mean.set_ylim(0, 0.62)


ax_DOM01_all = ph.add_twin(
    ax,
    color=conf_dict["DOM01"]["color"],
    labelcolor=conf_dict["DOM01"]["color"],
    width=2,
)
ax_DOM01_all.set_yticks([means["DOM01_overall_mean"]], [])
ax_DOM01_all.set_ylim(0, 0.62)

ax_DOM01_all_per = ph.add_twin(
    ax,
    color=conf_dict["DOM01"]["color"],
    labelcolor=conf_dict["DOM01"]["color"],
    width=1,
)
ax_DOM01_all_per.set_yticks(
    [means["DOM01_overall_25th"], means["DOM01_overall_75th"]], []
)
ax_DOM01_all_per.set_ylim(0, 0.62)

ax_DOM02_all = ph.add_twin(
    ax,
    color=conf_dict["DOM02"]["color"],
    labelcolor=conf_dict["DOM02"]["color"],
    width=2,
)
ax_DOM02_all.set_yticks([means["DOM02_overall_mean"]], [])
ax_DOM02_all.set_ylim(0, 0.62)

ax_DOM02_all_per = ph.add_twin(
    ax,
    color=conf_dict["DOM02"]["color"],
    labelcolor=conf_dict["DOM02"]["color"],
    width=1,
)
ax_DOM02_all_per.set_yticks(
    [means["DOM02_overall_25th"], means["DOM02_overall_75th"]], []
)
ax_DOM02_all_per.set_ylim(0, 0.62)


ax_obs_low_clouds = ax.twinx()
ax_obs_low_clouds.set_yticks(
    [np.round(obs_lowcloud_25th, 2), np.round(obs_lowcloud_75th, 2)]
)
ax_obs_low_clouds.set_ylim(0, 0.62)
ax_obs_low_clouds.set_yticks(np.arange(0, 0.62, 0.1), minor=True)

ax_obs_low_clouds_mean = ph.add_twin_right(
    ax, color="k", labelcolor="k", width=2, direction="out"
)
ax_obs_low_clouds_mean.set_yticks([np.round(obs_lowcloud_mean, 2)])
ax_obs_low_clouds_mean.set_ylim(0, 0.62)

ax_DOM01_lowclouds = ph.add_twin_right(
    ax,
    color=conf_dict["DOM01"]["color"],
    labelcolor=conf_dict["DOM01"]["color"],
    width=2,
)
ax_DOM01_lowclouds.set_yticks([means["DOM01_lowclouds_mean"]], [])
ax_DOM01_lowclouds.set_ylim(0, 0.62)

ax_DOM01_lowclouds_per = ph.add_twin_right(
    ax,
    color=conf_dict["DOM01"]["color"],
    labelcolor=conf_dict["DOM01"]["color"],
    width=1,
)
ax_DOM01_lowclouds_per.set_yticks(
    [means["DOM01_lowclouds_25th"], means["DOM01_lowclouds_75th"]], []
)
ax_DOM01_lowclouds_per.set_ylim(0, 0.62)

ax_DOM02_lowclouds = ph.add_twin_right(
    ax,
    color=conf_dict["DOM02"]["color"],
    labelcolor=conf_dict["DOM02"]["color"],
    width=2,
)
ax_DOM02_lowclouds.set_yticks([means["DOM02_lowclouds_mean"]], [])
ax_DOM02_lowclouds.set_ylim(0, 0.62)

ax_DOM02_lowclouds_per = ph.add_twin_right(
    ax,
    color=conf_dict["DOM02"]["color"],
    labelcolor=conf_dict["DOM02"]["color"],
    width=1,
)
ax_DOM02_lowclouds_per.set_yticks(
    [means["DOM02_lowclouds_25th"], means["DOM02_lowclouds_75th"]], []
)
ax_DOM02_lowclouds_per.set_ylim(0, 0.62)

axs.legend(frameon=False, bbox_to_anchor=(1.1, 1))

sns.despine(offset=10, right=False)
out_fn = f"FIG_CC_timeseries_{'_'.join(experiments)}.pdf"
out_path = os.path.join(fig_output_folder, out_fn)
out_dir = os.path.dirname(out_path)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
plt.savefig(out_path, bbox_inches="tight")
# -

obs_overall_mean

# +
fig, axs = plt.subplots(1, 1, figsize=(8, 2.2), dpi=300)
df_simulation = xr.open_dataset(
    cfg.ANALYSIS.MESOSCALE.METRICS.output_filename_fmt.format(
        type="rttov", DOM=2, exp=2
    )
)
df_simulation = df_simulation.isel(index=df_simulation.percentile_BT > 100)
times_sim = set(df_simulation.time.values)
times_obs = set(df_goes16_dom1.time.values)
common_times = sorted(times_obs.intersection(times_sim))
data_obs = df_goes16_dom1.set_index(index="time").sel(
    index=slice("2020-01-10", np.max(list(common_times)))
)
axs.plot(
    data_obs.index,
    data_obs.percentile_BT,
    label=conf_dict["obs"]["label"],
    color=conf_dict["obs"]["color"],
)

high_cloud_idx = np.where(data_obs.percentile_BT <= 290, 0.4, np.nan)
low_cloud_mask = np.where(data_obs.percentile_BT <= 290, False, True)
axs.plot(data_obs.index, high_cloud_idx, color="grey", alpha=0.2, linewidth=10)

obs_overall_mean = float(data_obs.percentile_BT.median())
obs_overall_25th = float(data_obs.percentile_BT.quantile(0.25))
obs_overall_75th = float(data_obs.percentile_BT.quantile(0.75))

obs_lowcloud_mean = float(data_obs.sel(index=low_cloud_mask).percentile_BT.median())
obs_lowcloud_25th = float(
    data_obs.sel(index=low_cloud_mask).percentile_BT.quantile(0.25)
)
obs_lowcloud_75th = float(
    data_obs.sel(index=low_cloud_mask).percentile_BT.quantile(0.75)
)

if __name__ == "__main__":
    logging.info(
        f"OBS: mean: {obs_overall_mean:.3f}, mean (without high clouds): {obs_lowcloud_mean:.3f}"
    )
times_obs = set(
    df_goes16_dom1.set_index(index="time")
    .sel(index=slice(np.min(list(common_times)), np.max(list(common_times))))
    .index.values
)
experiments = []
means = {}
for experiment in [2]:
    for domain in [1, 2]:
        experiments.append(f"exp{experiment}.dom{domain}")
        confs = conf_dict["DOM{DOM:02g}".format(DOM=domain)]
        try:
            df_simulation = xr.open_dataset(
                cfg.ANALYSIS.MESOSCALE.METRICS.output_filename_fmt.format(
                    type="rttov", DOM=domain, exp=experiment
                )
            )
        except FileNotFoundError:
            continue
        df_simulation = df_simulation.isel(index=df_simulation.percentile_BT > 100)
        times_sim = set(df_simulation.time.values)

        common_times = sorted(times_obs.intersection(times_sim))
        logging.debug("Resampling")
        data = (
            df_simulation.set_index(index="time")
            .percentile_BT.drop_duplicates("index")
            .sortby("index")
            .sel(index=sorted(common_times))
            .resample(index="10T")
            .nearest(tolerance="10T")
        )
        logging.debug("Plotting")
        axs.plot(
            data.index,
            data,
            label=confs["label"],
            color=confs["color"],
            alpha=alphas["exp" + str(experiment)],
        )

        data_reindex = data.reindex(
            index=data_obs.index, method="nearest", tolerance="5T"
        )
        #     print(data_reindex.sel(index=low_cloud_mask).mean())
        means[f"DOM0{domain}_overall_mean"] = float(data.median())
        means[f"DOM0{domain}_overall_25th"] = float(data.quantile(0.25))
        means[f"DOM0{domain}_overall_75th"] = float(data.quantile(0.75))
        means[f"DOM0{domain}_lowclouds_mean"] = float(
            data_reindex.sel(index=low_cloud_mask).median()
        )
        means[f"DOM0{domain}_lowclouds_25th"] = float(
            data_reindex.sel(index=low_cloud_mask).quantile(0.25)
        )
        means[f"DOM0{domain}_lowclouds_75th"] = float(
            data_reindex.sel(index=low_cloud_mask).quantile(0.75)
        )
        if __name__ == "__main__":
            logging.info(
                f"{descr[experiment]} "
                f'DOM0{domain}: mean: {means[f"DOM0{domain}_overall_mean"]},'
                f'mean (without high clouds): {means[f"DOM0{domain}_lowclouds_mean"]}'
            )

plt.ylabel(r"$T_{\mathrm{B, 25th}}$ / K")
plt.xlabel("date / UTC mm/dd")
plt.xlim(datetime.datetime(2020, 1, 12), datetime.datetime(2020, 2, 20))
axs.xaxis.set_major_formatter(md.DateFormatter(fmt="%m/%d"))
axs.set_ylim(240, 300)

ax = axs
ax.set_yticks([], labels=[])
ax.set_yticks(np.arange(240, 300, 10), minor=True)
ax_obs_all_mean = ph.add_twin(ax, color="k", labelcolor="k", width=2, direction="out")
ax_obs_all_mean.set_yticks([np.round(obs_overall_mean)])
ax_obs_all_mean.set_ylim(240, 300)


ax_DOM01_all = ph.add_twin(
    ax,
    color=conf_dict["DOM01"]["color"],
    labelcolor=conf_dict["DOM01"]["color"],
    width=2,
)
ax_DOM01_all.set_yticks([means["DOM01_overall_mean"]], [])
ax_DOM01_all.set_ylim(240, 300)

ax_DOM02_all = ph.add_twin(
    ax,
    color=conf_dict["DOM02"]["color"],
    labelcolor=conf_dict["DOM02"]["color"],
    width=2,
)
ax_DOM02_all.set_yticks([means["DOM02_overall_mean"]], [])
ax_DOM02_all.set_ylim(240, 300)

ax_obs_low_clouds = ax.twinx()
ax_obs_low_clouds.set_yticks([], labels=[])
ax_obs_low_clouds.set_ylim(240, 300)
ax_obs_low_clouds.set_yticks(np.arange(240, 300, 10), minor=True)

ax_obs_low_clouds_mean = ph.add_twin_right(
    ax, color="k", labelcolor="k", width=2, direction="out"
)
ax_obs_low_clouds_mean.set_yticks([np.round(obs_lowcloud_mean)])
ax_obs_low_clouds_mean.set_ylim(240, 300)

ax_DOM01_lowclouds = ph.add_twin_right(
    ax,
    color=conf_dict["DOM01"]["color"],
    labelcolor=conf_dict["DOM01"]["color"],
    width=2,
)
ax_DOM01_lowclouds.set_yticks([means["DOM01_lowclouds_mean"]], [])
ax_DOM01_lowclouds.set_ylim(240, 300)

ax_DOM02_lowclouds = ph.add_twin_right(
    ax,
    color=conf_dict["DOM02"]["color"],
    labelcolor=conf_dict["DOM02"]["color"],
    width=2,
)
ax_DOM02_lowclouds.set_yticks([means["DOM02_lowclouds_mean"]], [])
ax_DOM02_lowclouds.set_ylim(240, 300)

axs.hlines(
    290,
    datetime.datetime(2020, 1, 12),
    datetime.datetime(2020, 2, 20),
    color="lightgrey",
    zorder=0,
)

axs.legend(frameon=False, bbox_to_anchor=(1.1, 1))

sns.despine(offset=10, right=False)
out_fn = f"FIG_CC_brightness_temperature_{'_'.join(experiments)}.pdf"
out_path = os.path.join(fig_output_folder, out_fn)
out_dir = os.path.dirname(out_path)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
plt.savefig(out_path, bbox_inches="tight")

# +
ylims = (0, 0.201)
fig, axs = plt.subplots(1, 1, figsize=(8, 2.2), dpi=300)
df_simulation = xr.open_dataset(
    cfg.ANALYSIS.MESOSCALE.METRICS.output_filename_fmt.format(
        type="rttov", DOM=2, exp=2
    )
)
df_simulation = df_simulation.isel(index=df_simulation.percentile_BT > 100)
times_sim = set(df_simulation.time.values)

data_obs = df_goes16_dom1.set_index(index="time").sel(
    index=slice("2020-02-01", "2020-02-07")
)
times_obs = set(data_obs.index.values)
common_times = sorted(times_obs.intersection(times_sim))

axs.plot(
    data_obs.index,
    data_obs.cloud_fraction,
    label=conf_dict["obs"]["label"],
    color=conf_dict["obs"]["color"],
)

high_cloud_idx = np.where(data_obs.percentile_BT <= 290, 0.18, np.nan)
low_cloud_mask = np.where(data_obs.percentile_BT <= 290, False, True)
axs.plot(data_obs.index, high_cloud_idx, color="grey", alpha=0.2, linewidth=10)

obs_overall_mean = float(data_obs.cloud_fraction.median())
obs_overall_25th = float(data_obs.cloud_fraction.quantile(0.25))
obs_overall_75th = float(data_obs.cloud_fraction.quantile(0.75))

obs_lowcloud_mean = float(data_obs.sel(index=low_cloud_mask).cloud_fraction.median())
obs_lowcloud_25th = float(
    data_obs.sel(index=low_cloud_mask).cloud_fraction.quantile(0.25)
)
obs_lowcloud_75th = float(
    data_obs.sel(index=low_cloud_mask).cloud_fraction.quantile(0.75)
)


if __name__ == "__main__":
    logging.info(
        f"OBS: mean: {obs_overall_mean}, mean (without high clouds): {obs_lowcloud_mean}"
    )
times_obs = set(
    data_obs.index.sel(
        index=slice(np.min(list(common_times)), np.max(list(common_times)))
    ).index.values
)
experiments = []
means = {}
for experiment in [2]:
    for domain in [1, 2, 3]:
        experiments.append(f"exp{experiment}.dom{domain}")
        confs = conf_dict["DOM{DOM:02g}".format(DOM=domain)]
        try:
            df_simulation = xr.open_dataset(
                cfg.ANALYSIS.MESOSCALE.METRICS.output_filename_fmt.format(
                    type="rttov", DOM=domain, exp=experiment
                )
            )
            df_simulation["time"] = df_simulation.time.dt.floor("1s")
        except FileNotFoundError:
            requested_file = cfg.ANALYSIS.MESOSCALE.METRICS.output_filename_fmt
            logging.warning(
                f'File {requested_file.format(type="rttov",DOM=domain,exp=experiment)} not found'
            )
            continue
        df_simulation = df_simulation.isel(index=df_simulation.percentile_BT > 100)
        times_sim = set(df_simulation.time.values)

        common_times = sorted(times_obs.intersection(times_sim))
        logging.debug("Resampling")
        data = (
            df_simulation.set_index(index="time")
            .cloud_fraction.drop_duplicates("index")
            .sortby("index")
            .sel(index=sorted(common_times))
            .resample(index="10T")
            .nearest(tolerance="10T")
        )
        logging.debug("Plotting")
        axs.plot(
            data.index,
            data,
            label=confs["label"],
            color=confs["color"],
            alpha=alphas["exp" + str(experiment)],
        )

        data_reindex = data.reindex(
            index=data_obs.index, method="nearest", tolerance="5T"
        )
        #     print(data_reindex.sel(index=low_cloud_mask).mean())
        means[f"DOM0{domain}_overall_mean"] = float(data.median())
        means[f"DOM0{domain}_overall_25th"] = float(data.quantile(0.25))
        means[f"DOM0{domain}_overall_75th"] = float(data.quantile(0.75))
        means[f"DOM0{domain}_lowclouds_mean"] = float(
            data_reindex.sel(index=low_cloud_mask).median()
        )
        means[f"DOM0{domain}_lowclouds_25th"] = float(
            data_reindex.sel(index=low_cloud_mask).quantile(0.25)
        )
        means[f"DOM0{domain}_lowclouds_75th"] = float(
            data_reindex.sel(index=low_cloud_mask).quantile(0.75)
        )
        if __name__ == "__main__":
            logging.info(
                f"{descr[experiment]} "
                f'DOM0{domain}: mean: {means[f"DOM0{domain}_overall_mean"]:.3f},'
                f'mean (without high clouds): {means[f"DOM0{domain}_lowclouds_mean"]:.3f}'
            )

plt.ylabel(r"$C_{\mathrm{B}}$")
plt.xlabel("date / UTC mm/dd")
# plt.xlim(datetime.datetime(2020,1,12), datetime.datetime(2020,2,20))
axs.xaxis.set_major_formatter(md.DateFormatter(fmt="%m/%d"))
axs.set_ylim(ylims)

ax = axs
ax.set_yticks([np.round(obs_overall_25th, 2), np.round(obs_overall_75th, 2)])
ax.set_yticks(np.arange(*ylims, 0.1), minor=True)
ax_obs_all_mean = ph.add_twin(ax, color="k", labelcolor="k", width=2, direction="out")
ax_obs_all_mean.set_yticks([np.round(obs_overall_mean, 2)])
ax_obs_all_mean.set_ylim(ylims)


ax_DOM01_all = ph.add_twin(
    ax,
    color=conf_dict["DOM01"]["color"],
    labelcolor=conf_dict["DOM01"]["color"],
    width=2,
)
ax_DOM01_all.set_yticks([means["DOM01_overall_mean"]], [])
ax_DOM01_all.set_ylim(ylims)

ax_DOM01_all_per = ph.add_twin(
    ax,
    color=conf_dict["DOM01"]["color"],
    labelcolor=conf_dict["DOM01"]["color"],
    width=1,
)
ax_DOM01_all_per.set_yticks(
    [means["DOM01_overall_25th"], means["DOM01_overall_75th"]], []
)
ax_DOM01_all_per.set_ylim(ylims)

ax_DOM02_all = ph.add_twin(
    ax,
    color=conf_dict["DOM02"]["color"],
    labelcolor=conf_dict["DOM02"]["color"],
    width=2,
)
ax_DOM02_all.set_yticks([means["DOM02_overall_mean"]], [])
ax_DOM02_all.set_ylim(ylims)

ax_DOM02_all_per = ph.add_twin(
    ax,
    color=conf_dict["DOM02"]["color"],
    labelcolor=conf_dict["DOM02"]["color"],
    width=1,
)
ax_DOM02_all_per.set_yticks(
    [means["DOM02_overall_25th"], means["DOM02_overall_75th"]], []
)
ax_DOM02_all_per.set_ylim(ylims)

ax_DOM03_all = ph.add_twin(
    ax,
    color=conf_dict["DOM03"]["color"],
    labelcolor=conf_dict["DOM03"]["color"],
    width=2,
)
ax_DOM03_all.set_yticks([means["DOM03_overall_mean"]], [])
ax_DOM03_all.set_ylim(ylims)

ax_DOM03_all_per = ph.add_twin(
    ax,
    color=conf_dict["DOM03"]["color"],
    labelcolor=conf_dict["DOM03"]["color"],
    width=1,
)
ax_DOM03_all_per.set_yticks(
    [means["DOM03_overall_25th"], means["DOM03_overall_75th"]], []
)
ax_DOM03_all_per.set_ylim(ylims)


ax_obs_low_clouds = ax.twinx()
ax_obs_low_clouds.set_yticks(
    [np.round(obs_lowcloud_25th, 2), np.round(obs_lowcloud_75th, 2)]
)
ax_obs_low_clouds.set_ylim(ylims)
ax_obs_low_clouds.set_yticks(np.arange(*ylims, 0.1), minor=True)

ax_obs_low_clouds_mean = ph.add_twin_right(
    ax, color="k", labelcolor="k", width=2, direction="out"
)
ax_obs_low_clouds_mean.set_yticks([np.round(obs_lowcloud_mean, 2)])
ax_obs_low_clouds_mean.set_ylim(ylims)

ax_DOM01_lowclouds = ph.add_twin_right(
    ax,
    color=conf_dict["DOM01"]["color"],
    labelcolor=conf_dict["DOM01"]["color"],
    width=2,
)
ax_DOM01_lowclouds.set_yticks([means["DOM01_lowclouds_mean"]], [])
ax_DOM01_lowclouds.set_ylim(ylims)

ax_DOM01_lowclouds_per = ph.add_twin_right(
    ax,
    color=conf_dict["DOM01"]["color"],
    labelcolor=conf_dict["DOM01"]["color"],
    width=1,
)
ax_DOM01_lowclouds_per.set_yticks(
    [means["DOM01_lowclouds_25th"], means["DOM01_lowclouds_75th"]], []
)
ax_DOM01_lowclouds_per.set_ylim(ylims)

ax_DOM02_lowclouds = ph.add_twin_right(
    ax,
    color=conf_dict["DOM02"]["color"],
    labelcolor=conf_dict["DOM02"]["color"],
    width=2,
)
ax_DOM02_lowclouds.set_yticks([means["DOM02_lowclouds_mean"]], [])
ax_DOM02_lowclouds.set_ylim(ylims)

ax_DOM02_lowclouds_per = ph.add_twin_right(
    ax,
    color=conf_dict["DOM02"]["color"],
    labelcolor=conf_dict["DOM02"]["color"],
    width=1,
)
ax_DOM02_lowclouds_per.set_yticks(
    [means["DOM02_lowclouds_25th"], means["DOM02_lowclouds_75th"]], []
)
ax_DOM02_lowclouds_per.set_ylim(ylims)


ax_DOM03_lowclouds = ph.add_twin_right(
    ax,
    color=conf_dict["DOM03"]["color"],
    labelcolor=conf_dict["DOM03"]["color"],
    width=2,
)
ax_DOM03_lowclouds.set_yticks([means["DOM03_lowclouds_mean"]], [])
ax_DOM03_lowclouds.set_ylim(ylims)

ax_DOM03_lowclouds_per = ph.add_twin_right(
    ax,
    color=conf_dict["DOM03"]["color"],
    labelcolor=conf_dict["DOM03"]["color"],
    width=1,
)
ax_DOM03_lowclouds_per.set_yticks(
    [means["DOM03_lowclouds_25th"], means["DOM03_lowclouds_75th"]], []
)
ax_DOM03_lowclouds_per.set_ylim(ylims)


axs.legend(frameon=False, bbox_to_anchor=(1.1, 1))

sns.despine(offset=10, right=False)
out_fn = f"FIG_CC_timeseries_{'_'.join(experiments)}_dom03_timeslice.pdf"
out_path = os.path.join(fig_output_folder, out_fn)
out_dir = os.path.dirname(out_path)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
plt.savefig(out_path, bbox_inches="tight")
