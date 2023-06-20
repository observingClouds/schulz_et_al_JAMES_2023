# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all
#     formats: ipynb,py:light
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
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

# +
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from omegaconf import OmegaConf

cfg = OmegaConf.load("../config/paths.cfg")
params = OmegaConf.load("../config/mesoscale_params.yaml")
simulation_params = OmegaConf.load("../config/simulation.yaml")

threshold_freq = params.manual_classifications.threshold_pattern
start_analysis = simulation_params.ICON312m.dates.start_analysis
stop_analysis = simulation_params.ICON312m.dates.stop_analysis
dates_of_interest = pd.date_range(start_analysis, stop_analysis, freq="1D")

ds_max_fn = cfg.ANALYSIS.MESOSCALE.CLASSIFICATIONS.manual.IR.classes
ds_max = xr.open_dataset(ds_max_fn)
ds_max = ds_max.sel(date=dates_of_interest)


color_dict = {
    "Sugar": "#A1D791",
    "Flower": "#93D2E2",
    "Fish": "#2281BB",
    "Gravel": "#3EAE47",
    "Flowers": "#93D2E2",
    "Unclassified": "lightgrey",
}


# +
max_freq = ds_max["max_freq"]
max_pattern = ds_max["max_pattern"]
mean_pattern_freq = ds_max["mean_freq"]

plt.figure(figsize=(8, 0.5), dpi=100)
plt.bar(
    mean_pattern_freq.date.values.astype("datetime64[h]"),
    np.ones(len(max_freq)),
    color="lightgrey",
)
for p in sorted(np.unique(max_pattern)):
    p_dates = np.where(max_pattern == p)[0]
    pattern = mean_pattern_freq.pattern.values[p]
    label = pattern
    if pattern == "Unclassified":
        label = "Mixed"
    plt.bar(
        mean_pattern_freq.date[p_dates].values.astype("datetime64[h]"),
        (max_freq[p_dates] > threshold_freq),
        color=color_dict[pattern],
        label=label,
    )
plt.ylim(0, 1)
plt.xticks(rotation=-45)
plt.tick_params(left=False, labelleft=False)
plt.legend(ncol=5, loc="center", bbox_to_anchor=(0.5, -3, 0, 0), frameon=False)
sns.despine(left=True)
plt.savefig(
    "../figures/Patterns_manualClassifications_mostCommon.pdf", bbox_inches="tight"
)
# -
