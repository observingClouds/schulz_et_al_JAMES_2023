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

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from intake import open_catalog

cat_address = (
    "https://raw.githubusercontent.com/eurec4a/eurec4a-intake/master/catalog.yml"
)
cat = open_catalog(cat_address)

conf_dict = {
    "DOM01": {"label": "ICON 624m", "color": "#109AFA"},
    "DOM02": {"label": "ICON 312m", "color": "red"},
    "obs": {"label": "GOES-16 ABI", "color": "black"},
}

radar_ds = cat.barbados.bco.radar_reflectivity.to_dask()

# + tags=[]
# Daily means
resampling_histograms = {}

station = "BCO"
heights = [300, 750, 1500]  # radar_ds.sel(range=slice(150, 750)).range.values
bins = np.arange(-70, 40, 2)

## Simulations
for height in heights:
    for experiment in ["control"]:  # ,'highCCN']:
        for domain in [1, 2]:
            print(domain)
            pamtra_ds = cat.simulations.ICON[f"LES_CampaignDomain_{experiment}"][
                f"synthetic_radar_{station}_DOM0{domain}"
            ].to_dask()
            common_times = sorted(
                set(radar_ds.time.dt.floor("1T").values).intersection(
                    pamtra_ds.time.values
                )
            )
            resampling_profile = (
                pamtra_ds.sel(time=common_times)
                .sel(height=height, method="nearest")
                .Z_att
            )
            resampling_histograms[experiment, domain, height] = np.histogram(
                resampling_profile, bins=bins
            )
    radar_resampled = (
        radar_ds.Zf.resample(time="1T")
        .nearest(tolerance="1s")
        .sel(time=common_times)
        .sel(range=height, method="nearest")
    )
    resampling_histograms["radar", 0, height] = np.histogram(radar_resampled, bins=bins)
# -

histograms_flattened = {}
for key in resampling_histograms.keys():
    arr = np.zeros(10000)  # resampling_histograms[key][0].sum()+1000)
    arr[:] = np.nan
    bins = resampling_histograms[key][1][:-1]
    counts = resampling_histograms[key][0]
    i = 0
    for b, bi in enumerate(bins):
        count = counts[b]
        arr[i : i + count] = bi
        i += count
    histograms_flattened[key] = arr

fig, axs = plt.subplots(1, len(heights), sharex=True, sharey=True, figsize=(8, 2))
for h, height in enumerate(heights):
    axs[h].stairs(
        resampling_histograms[("radar", 0, height)][0]
        / np.sum(resampling_histograms[("radar", 0, height)][0])
        * 100,
        resampling_histograms[("radar", 0, height)][1],
        color=conf_dict["obs"]["color"],
        label="observations",
    )
    axs[h].stairs(
        resampling_histograms[("control", 1, height)][0]
        / np.sum(resampling_histograms[("control", 1, height)][0])
        * 100,
        resampling_histograms[("control", 1, height)][1],
        color=conf_dict["DOM01"]["color"],
        label="ICON-624m",
    )
    axs[h].stairs(
        resampling_histograms[("control", 2, height)][0]
        / np.sum(resampling_histograms[("control", 1, height)][0])
        * 100,
        resampling_histograms[("control", 2, height)][1],
        color=conf_dict["DOM02"]["color"],
        label="ICON-312m",
    )

    axs[h].set_title(f"{height}m")
    axs[h].set_xlabel("dBZ")
    axs[h].set_xlim(min(bins), max(bins))
axs[0].set_ylabel("freq. of occurrence / %")
plt.legend(ncol=3, bbox_to_anchor=(0.5, -0.3))
# plt.tight_layout()
sns.despine()
plt.savefig("../figures/echofraction_distribution_combined.pdf", bbox_inches="tight")
