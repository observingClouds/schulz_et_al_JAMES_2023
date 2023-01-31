"""Defining mesoscale groups.

In order to split the EUREC4A timeseries in different pieces with
similar meso-scale organizations that can be analyzed, each day (or even
a subset of this) needs to be classified.
"""
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from intake import open_catalog
from omegaconf import OmegaConf

cfg = OmegaConf.load("../../config/paths.cfg")
params = OmegaConf.load("../../config/mesoscale_params.yaml")

cat = open_catalog(
    "https://raw.githubusercontent.com/eurec4a/eurec4a-intake/master/catalog.yml"
)

output_classifications = cfg.ANALYSIS.MESOSCALE.CLASSIFICATIONS.manual.IR.classes
output_mostcommon_classifications = (
    cfg.ANALYSIS.MESOSCALE.CLASSIFICATIONS.manual.IR.class_decision
)

geobounds = {}
geobounds["lat_min"] = params.metrics.geobounds.lat_min
geobounds["lat_max"] = params.metrics.geobounds.lat_max
geobounds["lon_min"] = params.metrics.geobounds.lon_min
geobounds["lon_max"] = params.metrics.geobounds.lon_max
threshold_freq = params.manual_classifications.threshold_pattern

color_dict = {
    "Sugar": "#A1D791",
    "Flower": "#93D2E2",
    "Fish": "#2281BB",
    "Gravel": "#3EAE47",
    "Flowers": "#93D2E2",
    "Unclassified": "grey",
}

manual_classifications_ds = cat.c3ontext.level3_IR_daily.to_dask()

mean_pattern_freq = (
    manual_classifications_ds.sel(
        latitude=slice(geobounds["lat_max"], geobounds["lat_min"]),
        longitude=slice(geobounds["lon_min"], geobounds["lon_max"]),
    )
    .freq.mean(["latitude", "longitude"])
    .fillna(0)
)

# What is the most common pattern each day, with an "agreement" of at least X
max_pattern = mean_pattern_freq.argmax("pattern")
max_freq = mean_pattern_freq.max("pattern")

ds_max = xr.Dataset()
ds_max["max_pattern"] = max_pattern
ds_max["max_freq"] = max_freq
ds_max["mean_freq"] = mean_pattern_freq
ds_max.to_netcdf(
    output_classifications.format(geobounds=geobounds),
    encoding={"pattern": {"dtype": "S1"}},
)

# JSON version
dict_pattern = {}
for d, date in enumerate(mean_pattern_freq.date.values):
    if max_freq[d] > threshold_freq:
        dict_pattern[date] = str(mean_pattern_freq.pattern[int(max_pattern[d])].values)
    else:
        dict_pattern[date] = "Mixed"
df_pattern = pd.DataFrame.from_dict(dict_pattern, orient="index", columns=["pattern"])
if not os.path.exists(os.path.dirname(output_mostcommon_classifications)):
    os.makedirs(os.path.dirname(output_mostcommon_classifications))
df_pattern.to_json(output_mostcommon_classifications, date_format="iso")
