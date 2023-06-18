#!/work/mh0010/m300408/envs/covariability/bin/python
# -*- coding: utf-8 -*-

# Calculate mesoscale metrics for synthetic satellite data

# SBATCH --account=mh0010
# SBATCH --job-name=mesoscalemetrics
# SBATCH --partition=gpu
# SBATCH --nodes=1
# SBATCH --threads-per-core=2
# SBATCH --output=logs/LOG.mesoscale_metrics.%j.o
# SBATCH --error=logs/LOG.mesoscale_metrics.%j.o
# SBATCH --exclusive
# SBATCH --chdir=/work/mh0010/m300408/covariability_les_obs/analysis/mesoscale
# SBATCH --time=08:00:00
# SBATCH --mail-user=hauke.schulz@mpimet.mpg.de
# SBATCH --mail-type=ALL


import sys

import numpy as np
import pandas as pd
import tqdm
import xarray as xr
from intake import open_catalog
from omegaconf import OmegaConf

sys.path.append("../../src/deps/")
from coat import organization as org  # noqa: E402

cfg = OmegaConf.load("../../config/paths.cfg")
params = OmegaConf.load("../../config/mesoscale_params.yaml")

cat = open_catalog(
    "https://raw.githubusercontent.com/eurec4a/eurec4a-intake/master/catalog.yml"
)

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

experiment = 2
exp_longname = "LES_CampaignDomain_control"
for domain in [1, 2, 3]:
    metrics_output_filename_fmt = cfg.ANALYSIS.MESOSCALE.METRICS.output_filename_fmt
    sysnsat_input_filename_fmt = cat.simulations.ICON[exp_longname][
        f"rttov_DOM0{domain}"
    ]

    ds_rttov = sysnsat_input_filename_fmt.to_dask()

    results_simulation = {}
    i = 0
    for t, time in enumerate(tqdm.tqdm(ds_rttov.time)):
        data = ds_rttov[
            "synsat_rttov_forward_model_{DOM}__abi_ir__goes_16__channel_7".format(
                DOM=domain
            )
        ].isel(time=t)
        data = data.sel(
            {
                "lat": slice(geobounds["lat_min"], geobounds["lat_max"]),
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
        print(time.values, Iorg, cloud_fraction)
        results_simulation[i] = {
            "domain": domain,
            "filename": sysnsat_input_filename_fmt.urlpath,
            "time": time.values,
            "Iorg": Iorg,
            "cluster_size_mean": mean_cluster_size,
            "cluster_size_std": std_cluster_size,
            "cluster_N": N,
            "cloud_fraction": cloud_fraction,
            "percentile_BT": percentile_data,
        }
        i += 1
    df_simulation = pd.DataFrame.from_dict(results_simulation, orient="index")
    df_simulation.to_xarray().to_netcdf(
        metrics_output_filename_fmt.format(DOM=domain, type="rttov", exp=experiment)
    )
