#!/work/mh0010/m300408/envs/covariability/bin/python
# -*- coding: utf-8 -*-

# Calculate mesoscale metrics for synthetic satellite data

# SBATCH --account=mh0010
# SBATCH --job-name=mesoscalemetrics
# SBATCH --partition=compute
# SBATCH --nodes=1
# SBATCH --threads-per-core=2
# SBATCH --output=logs/LOG.mesoscale_metrics.%j.o
# SBATCH --error=logs/LOG.mesoscale_metrics.%j.o
# SBATCH --exclusive
# SBATCH --chdir=/work/mh0010/m300408/covariability_les_obs/analysis/mesoscale
# SBATCH --time=00:20:00
# SBATCH --mail-user=hauke.schulz@mpimet.mpg.de
# SBATCH --mail-type=ALL


import os
import sys

import numpy as np
import pandas as pd
import tqdm
import xarray as xr
from omegaconf import OmegaConf

sys.path.append("../deps/")
from coat import organization as org  # noqa: E402

cfg = OmegaConf.load("../../config/paths.cfg")

metrics_output_filename_fmt = cfg.ANALYSIS.MESOSCALE.METRICS.output_filename_fmt
sat_input_filename_fmt = cfg.OBS.SATELLITES["GOES16"].CH13.filename_fmt

params = OmegaConf.load("../../config/mesoscale_params.yaml")
geobounds = {}
geobounds["lat_min"] = params.metrics.geobounds.lat_min
geobounds["lat_max"] = params.metrics.geobounds.lat_max
geobounds["lon_min"] = params.metrics.geobounds.lon_min
geobounds["lon_max"] = params.metrics.geobounds.lon_max
org_settings = {}
org_settings[
    "threshold_discard_percentile"
] = params.metrics.BTbounds.threshold_discard_percentile
org_settings[
    "threshold_discard_temperature"
] = params.metrics.BTbounds.threshold_discard_temperature
org_settings[
    "threshold_cluster_llimit"
] = params.metrics.BTbounds.threshold_cluster_llimit
org_settings[
    "threshold_cluster_ulimit"
] = params.metrics.BTbounds.threshold_cluster_ulimit
org_settings["stencil"] = np.ones(
    (params.metrics.iorg.stencil[0], params.metrics.iorg.stencil[1])
)

for domain in [1, 2]:
    dates = pd.date_range(
        params.metrics.dates.start,
        params.metrics.dates.stop,
        freq=params.metrics.dates.step,
    )

    results_satellite = {}
    i = 0
    for date in tqdm.tqdm(dates):
        f = date.strftime(sat_input_filename_fmt.format(dom=domain))
        if not os.path.exists(f):
            continue
        ds = xr.open_dataset(f)
        data = ds.C13.squeeze()
        data = data.sel(
            {
                "lat": slice(geobounds["lat_max"], geobounds["lat_min"]),
                "lon": slice(geobounds["lon_min"], geobounds["lon_max"]),
            }
        )
        (
            Iorg,
            mean_cluster_size,
            std_cluster_size,
            N,
            cloud_fraction,
            percentile_data,
        ) = org.get_organization_info(data, org_settings, verbose=False)
        print(f.split("/")[-1], date, Iorg, cloud_fraction)
        results_satellite[i] = {
            "domain": domain,
            "filename": f,
            "time": date,
            "Iorg": Iorg,
            "cluster_size_mean": mean_cluster_size,
            "cluster_size_std": std_cluster_size,
            "cluster_N": N,
            "cloud_fraction": cloud_fraction,
            "percentile_BT": percentile_data,
        }
        i += 1
    df = pd.DataFrame.from_dict(results_satellite, orient="index")
    df.to_xarray().to_netcdf(
        metrics_output_filename_fmt.format(DOM=domain, type="goes16", exp="")
    )
