"""Preprocessing for figure 6."""
import logging
import os
import sys

import fsspec
import matplotlib.pyplot as plt
import metpy.calc as mcalc
import metpy.units as munits
import numpy as np
import pandas as pd
import pint_xarray
import seaborn as sns
import tqdm
import xarray as xr
import zarr
from intake import open_catalog
from omegaconf import OmegaConf

sys.path.append("../helpers/")
sys.path.append("/home/m/m300408/GitProjects/Thermodynamics")
import aes_thermo as thermo  # noqa: E402
import cluster_helpers as ch  # noqa: E402
import grid_helpers as gh  # noqa: E402
import plotting as ph  # noqa: E40

params = OmegaConf.load("../../config/mesoscale_params.yaml")

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler("../../logs/tradewind_BL_stats.log"),
        logging.StreamHandler(),
    ],
)

if __name__ == "__main__":
    client = ch.setup_cluster("local cluster", verbose=logging.ERROR)
    logging.info(client)

    dom = 2
    include_obs = True
    visualize_cf_thetal_dependency = False
    days_with_dropsondes_only = False
    local_time = True
    overwrite = True

    # Region subset
    geobounds = {}
    geobounds["lat_min"] = params.metrics.geobounds.lat_min
    geobounds["lat_max"] = params.metrics.geobounds.lat_max
    geobounds["lon_min"] = params.metrics.geobounds.lon_min
    geobounds["lon_max"] = params.metrics.geobounds.lon_max

    lons = [geobounds["lon_min"], geobounds["lon_max"]]
    lats = [geobounds["lat_min"], geobounds["lat_max"]]

    circle_only = True
    center_pos = (13.3, -57.717)
    radius = 1  # deg

    tape_catalog_url = (
        "https://github.com/observingClouds/tape_archive_index/blob/main/catalog.yml"
    )
    online_catalog_url = (
        "https://raw.githubusercontent.com/observingClouds/"
        "eurec4a-intake/ICON-LES-control-DOM03/catalog.yml"
    )

    special_cases = {
        "Flowers": {"date": "2020-02-02 00:00:00", "color": "#93D2E2"},
        "Gravel": {"date": "2020-01-12 00:00:00", "color": "#3EAE47"},
        "Fish": {"date": "2020-01-22 00:00:00", "color": "#2281BB"},
        "Sugar": {"date": "2020-02-06 00:00:00", "color": "#A1D791"},
    }

    ## Loading data sources
    # Opening online catalog
    cat = open_catalog(online_catalog_url)

    # Load 3D dataset from swift resource or tape depending on availability
    try:
        ds = cat.simulations.ICON.LES_CampaignDomain_control[f"3D_DOM0{dom}"].to_dask()
    except BaseException:
        tape_cat = open_catalog(tape_catalog_url)
        ds = tape_cat[
            f"EUREC4A_ICON-LES_control_DOM0{dom}_3D_native"
        ].to_dask()  # not yet in catalog

    # Load horizontal grid
    grid = cat.simulations.grids[ds.uuidOfHGrid].to_dask()

    # Load vertical grid
    if dom == 2 or dom == 3:
        v_grid = cat.simulations.grids["ecf22d17-dcee-1510-a807-11ae4a612be0"].to_dask()
    else:
        v_grid = cat.simulations.grids[ds.uuidOfVGrid].to_dask()

    # Get geometric height from random ocean cell
    random_representative_cell = 100000
    assert v_grid.isel(cell=random_representative_cell).fr_land == 0
    assert v_grid.isel(cell=random_representative_cell).fr_lake == 0
    heights = v_grid.isel(cell=random_representative_cell).z_ifc

    ds["height"] = heights.values[ds.height.astype(int)]
    ds["height"] = (
        ds["height"].pint.quantify("m").pint.to("km").pint.dequantify().compute()
    )

    # Prepare dropsonde observations
    if include_obs:
        dropsondes = cat.dropsondes.JOANNE.level3.to_dask().swap_dims(
            {"sonde_id": "launch_time"}
        )
        dropsondes = dropsondes.sel(launch_time=dropsondes.platform_id == "HALO")
        θ_l_dropsonde = thermo.get_theta_l(dropsondes.ta, dropsondes.p, dropsondes.q)

    # Minor post-processing
    if dom == 1 or dom == 3:
        ds["qt"] = ds.qc + ds.qv + ds.qr
    elif dom == 2:
        ds["qt"] = ds.qc + ds.qv
    ds["qt"] = ds["qt"].pint.quantify("kg/kg").pint.to("g/kg").pint.dequantify()

    ## Select region
    if circle_only:
        mask_circle = gh.get_cells_within_circle(grid, center_pos, radius)
        cell_mask = mask_circle
    else:
        cell_mask, _ = gh.grid_selection(grid, lats=lats, lons=lons)

    ds_sel = ds.sel(time=slice("2020-01-11", "2020-02-18"), cell=cell_mask)

    ### Check selection visually
    fig = ph.plot(
        ds_sel.temp.sel(height=0, method="nearest").isel(time=20),
        grid.sel(cell=cell_mask),
        297,
        300,
        dpi=70,
    )
    plt.savefig(
        "../../figures/area_selected_for_tradewind_BL_statistics.pdf",
        bbox_inches="tight",
    )

    ## Calculate liquid water potential temperature and it's gradient
    theta_grad_fn = "../../data/intermediate/theta_l_gradient.zarr"

    if not os.path.exists(theta_grad_fn) or overwrite:
        θ_l = thermo.get_theta_l(ds_sel.temp, ds_sel.pres, ds_sel.qt / 1000)
        ds_sel["θₗ"] = θ_l
        grad = (
            ds_sel["θₗ"].isel(height=slice(0, -1)).data
            - ds_sel["θₗ"].isel(height=slice(1, None)).data
        )
        ds_sel["θl_gradient"] = xr.DataArray(
            grad,
            dims=("time", "height", "cell"),
            coords={
                "height": ds_sel["θₗ"].isel(height=slice(1, None)).height,
                "time": ds_sel.time,
            },
        )
        store = zarr.DirectoryStore(theta_grad_fn, dimension_separator="/")
        ds_sel.chunk({"time": 1, "height": 12, "cell": 505202})[
            ["θl_gradient", "θₗ"]
        ].to_zarr(store, mode="w")
        ds_theta_gradient = xr.open_zarr(theta_grad_fn)

    logging.info("Load dataset")
    ds_theta_gradient = xr.open_zarr(theta_grad_fn)
    ds_sel["θₗ"] = ds_theta_gradient["θₗ"]
    ds_sel["θl_gradient"] = ds_theta_gradient["θl_gradient"]

    hgt_sel = ds_sel["θl_gradient"].sel(height=slice(4, 0.8))
    max_grad_idx = (
        hgt_sel.where(~np.isnan(hgt_sel), -999)
        .argmax(dim="height", skipna=True)
        .compute()
    )
    max_grad_height = (
        ds_sel["θl_gradient"].sel(height=slice(4, 0.8)).height[max_grad_idx]
    )
    quantiles = max_grad_height.quantile([0.25, 0.5, 0.75])

    logging.info(f"Max. gradient height quantiles {quantiles}")
    logging.info(f"Median of max. gradient height {max_grad_height.median()}")
    logging.info("Median height by pattern:")
    for case, props in special_cases.items():
        max_grad_date = max_grad_height.groupby(
            max_grad_height.time.dt.floor("1D")
        ).median(["time", "cell"])
        max_grad_pattern = max_grad_date.sel(floor=props["date"]).values
        logging.info(f"{case} ({props['date']}): {max_grad_pattern}")

    logging.info("Calc cloud fraction")
    cf_mean = (
        (ds_sel[["qc"]] > 0)
        .groupby(ds_sel.time.dt.floor("1D"))
        .mean(["time", "cell"])
        .chunk({"floor": -1})
        .compute()
    )

    logging.info("Calc means over non-zero elements")
    # This has especially an effect on `qc` which is only defined in clouds
    means_nonzero = (
        ds_sel[["qt", "qc", "θₗ"]]
        .where(ds_sel > 0)
        .groupby(ds_sel.time.dt.floor("1D"))
        .mean(["time", "cell"])
        .chunk({"floor": -1})
        .compute()
    )

    logging.info("Calc mean of windspeed components")
    wspd_component_mean = (
        ds_sel[["u", "v"]]
        .groupby(ds_sel.time.dt.floor("1D"))
        .mean(["time", "cell"])
        .chunk({"floor": -1})
        .compute()
    )

    medians = means_nonzero

    if days_with_dropsondes_only:
        common_dates = sorted(
            set(dropsondes.launch_time.dt.date.values).intersection(
                medians.floor.dt.date.values
            )
        )
        medians = medians.sel(floor=np.in1d(medians.floor.dt.date.values, common_dates))
        logging.info(common_dates)

    data = mcalc.wind_speed(
        wspd_component_mean.u * munits.units["m/s"],
        wspd_component_mean.v * munits.units["m/s"],
    ).pint.dequantify()

    logging.info("Combining computed datasets")
    ds_out = xr.merge(
        [medians, cf_mean.rename({"qc": "cf"}), data.to_dataset(name="wspd")]
    )
    ds_out.to_netcdf("../../data/result/tradewind_BL_profiles.nc")
