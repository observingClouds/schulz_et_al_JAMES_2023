import numpy as np
import pandas as pd
import xarray as xr
from intake import open_catalog
from omegaconf import OmegaConf

cfg = OmegaConf.load("./config/paths.cfg")
params = OmegaConf.load("./config/mesoscale_params.yaml")
cat_tmp = open_catalog(
    "https://raw.githubusercontent.com/eurec4a/eurec4a-intake/master/catalog.yml"
)

valid_cells_limit = 0.9

df_goes16_dom1 = xr.open_dataset(
    cfg.ANALYSIS.MESOSCALE.METRICS.output_filename_fmt.format(
        type="goes16", DOM=1, exp="_"
    )[:-4]
    + ".nc"
)
df_rttov_dom1 = xr.open_dataset(
    cfg.ANALYSIS.MESOSCALE.METRICS.output_filename_fmt.format(type="rttov", DOM=1, exp=2)
)
df_rttov_dom2 = xr.open_dataset(
    cfg.ANALYSIS.MESOSCALE.METRICS.output_filename_fmt.format(type="rttov", DOM=2, exp=2)
)

print("High cloud occurrance")
count_highcloud_occurrance_DOM01 = (
    (df_rttov_dom1.valid_cells / df_rttov_dom1.valid_cells.max() < valid_cells_limit)
    .resample(time="1D")
    .sum()
)
days_without_highclouds_DOM01 = count_highcloud_occurrance_DOM01.where(
    count_highcloud_occurrance_DOM01 == 0, drop=True
).time
count_highcloud_occurrance_DOM02 = (
    (df_rttov_dom2.valid_cells / df_rttov_dom2.valid_cells.max() < valid_cells_limit)
    .resample(time="1D")
    .sum()
)
days_without_highclouds_DOM02 = count_highcloud_occurrance_DOM02.where(
    count_highcloud_occurrance_DOM02 == 0, drop=True
).time
count_highcloud_occurrance_CERES = (
    (df_goes16_dom1.valid_cells / df_goes16_dom1.valid_cells.max() < valid_cells_limit)
    .resample(time="1D")
    .sum()
)
days_without_highclouds_CERES = count_highcloud_occurrance_CERES.where(
    count_highcloud_occurrance_CERES == 0, drop=True
).time

print("Write output")
DOM02_CERES_common_days_without_highclouds = list(
    set(days_without_highclouds_CERES.values).intersection(
        days_without_highclouds_DOM02.values
    )
)

pd.DataFrame(
    DOM02_CERES_common_days_without_highclouds, columns=["no_high_cloud"]
).to_parquet("./data/result/no_high_clouds_DOM02.pq")

DOM01_CERES_common_days_without_highclouds = list(
    set(days_without_highclouds_CERES.values).intersection(
        days_without_highclouds_DOM01.values
    )
)
pd.DataFrame(
    DOM01_CERES_common_days_without_highclouds, columns=["no_high_cloud"]
).to_parquet("./data/result/no_high_clouds_DOM01.pq")
