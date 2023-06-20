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

# # Environmental condition

# ![image.png](attachment:a7ea0412-c36c-4d8e-be73-c2caf7644cc8.png)

# ## Total water specific humidity

# +
import logging
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

sys.path.append("/home/m/m300408/GitProjects/Thermodynamics")

import aes_thermo as thermo  # noqa: E402

sys.path.append("../src/helpers/")
import cluster_helpers as ch  # noqa: E402
import grid_helpers as gh  # noqa: E402
import plotting as ph  # noqa: E402

if __name__ == "__main__":  # noqa: C901
    dom = 2
    include_obs = True
    visualize_cf_thetal_dependency = False
    days_with_dropsondes_only = True
    local_time = True
    theta_grad_fn = "../data/intermediate/theta_l_gradient.zarr"
    BL_profiles_fn = "../data/result/tradewind_BL_profiles.nc"

    params = OmegaConf.load("../config/mesoscale_params.yaml")

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
        "https://raw.githubusercontent.com/observingClouds"
        "/eurec4a-intake/ICON-LES-control-DOM03/catalog.yml"
    )
    # -

    special_cases = {
        "Flowers": {"date": "2020-02-02 00:00:00", "color": "#93D2E2"},
        "Gravel": {"date": "2020-01-12 00:00:00", "color": "#3EAE47"},
        "Fish": {"date": "2020-01-22 00:00:00", "color": "#2281BB"},
        "Sugar": {"date": "2020-02-06 00:00:00", "color": "#A1D791"},
    }

    if circle_only:
        workflow_id = f"DOM0{dom}_{center_pos}+{radius}"
    else:
        workflow_id = f"DOM0{dom}_{lats},{lons}"

    # -

    client = ch.setup_cluster("local cluster", verbose=logging.ERROR)
    client

    # ### Loading data sources

    ds_out = xr.open_dataset(BL_profiles_fn)
    medians = ds_out

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

    # +
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
    # -

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

    ds

    # ### Select region

    if circle_only:
        mask_circle = gh.get_cells_within_circle(grid, center_pos, radius)
        cell_mask = mask_circle
    else:
        cell_mask, _ = gh.grid_selection(grid, lats=lats, lons=lons)

    ds_sel = ds.sel(time=slice("2020-01-11", "2020-02-18"), cell=cell_mask)

    print("Load dataset")
    ds_theta_gradient = xr.open_zarr(theta_grad_fn)
    ds_sel["θₗ"] = ds_theta_gradient["θₗ"]
    ds_sel["θl_gradient"] = ds_theta_gradient["θl_gradient"]
    # -

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

    fig = plt.figure(figsize=(3, 4), dpi=200)
    plt.fill_betweenx(
        medians.height.sel(height=slice(5, 0)),
        medians["θₗ"].sel(height=slice(5, 0)).quantile(0.25, dim="floor"),
        medians["θₗ"].sel(height=slice(5, 0)).quantile(0.75, dim="floor"),
        color="grey",
        alpha=0.2,
        linewidth=0,
    )
    plt.fill_betweenx(
        medians.height.sel(height=slice(5, 0)),
        medians["θₗ"].sel(height=slice(5, 0)).quantile(0, dim="floor"),
        medians["θₗ"].sel(height=slice(5, 0)).quantile(1, dim="floor"),
        color="grey",
        alpha=0.2,
        linewidth=0,
    )
    print("Going through cases")
    for _, case_props in special_cases.items():
        try:
            plt.plot(
                medians["θₗ"].sel(height=slice(5, 0)).sel(floor=case_props["date"]).T,
                medians.height.sel(height=slice(5, 0)),
                color=case_props["color"],
                alpha=1,
                linewidth=1,
            )
        except KeyError:
            continue
    print("Plotting median")
    plt.plot(
        medians["θₗ"].sel(height=slice(5, 0)).median(dim="floor").T,
        medians.height.sel(height=slice(5, 0)),
        color="black",
        alpha=1,
        linewidth=1,
    )
    if include_obs:
        plt.plot(
            θ_l_dropsonde.sel(alt=slice(0, 5000)).median(dim="launch_time"),
            θ_l_dropsonde.sel(alt=slice(0, 5000)).alt / 1000,
            color="black",
            alpha=1,
            linewidth=1,
            linestyle="--",
        )
    sns.despine()
    plt.savefig(f"../figures/profiles_3D_θₗ_{workflow_id}_mean.pdf", bbox_inches="tight")

    fig = plt.figure(figsize=(3, 4), dpi=200)
    plt.fill_betweenx(
        medians.height.sel(height=slice(5, 0)),
        medians["qt"].sel(height=slice(5, 0)).quantile(0.25, dim="floor"),
        medians["qt"].sel(height=slice(5, 0)).quantile(0.75, dim="floor"),
        color="grey",
        alpha=0.2,
        linewidth=0,
    )
    plt.fill_betweenx(
        medians.height.sel(height=slice(5, 0)),
        medians["qt"].sel(height=slice(5, 0)).quantile(0, dim="floor"),
        medians["qt"].sel(height=slice(5, 0)).quantile(1, dim="floor"),
        color="darkgrey",
        alpha=0.2,
        linewidth=0,
    )
    for _, case_props in special_cases.items():
        try:
            plt.plot(
                medians["qt"].sel(height=slice(5, 0)).sel(floor=case_props["date"]).T,
                medians.height.sel(height=slice(5, 0)),
                color=case_props["color"],
                alpha=1,
                linewidth=1,
            )
        except KeyError:
            continue
    plt.plot(
        medians["qt"].sel(height=slice(5, 0)).median(dim="floor").T,
        medians.height.sel(height=slice(5, 0)),
        color="black",
        alpha=1,
        linewidth=1,
    )
    if include_obs:
        plt.plot(
            dropsondes.sel(alt=slice(10, 5000)).q.median(dim="launch_time") * 1000,
            dropsondes.sel(alt=slice(10, 5000)).alt / 1000,
            color="black",
            alpha=1,
            linewidth=1,
            linestyle="--",
        )
    sns.despine()
    plt.savefig(f"../figures/profiles_3D_qt_{workflow_id}_mean.pdf", bbox_inches="tight")

    var = "qc"
    fig = plt.figure(figsize=(3, 4), dpi=200)
    medians["qc"] = medians["qc"].where(~np.isnan(medians["qc"]), 0)
    plt.fill_betweenx(
        medians.height.sel(height=slice(5, 0)),
        medians[var].sel(height=slice(5, 0)).quantile(0.25, dim="floor") * 1000,
        medians[var].sel(height=slice(5, 0)).quantile(0.75, dim="floor") * 1000,
        color="grey",
        alpha=0.2,
        linewidth=0,
    )
    plt.fill_betweenx(
        medians.height.sel(height=slice(5, 0)),
        medians[var].sel(height=slice(5, 0)).quantile(0, dim="floor") * 1000,
        medians[var].sel(height=slice(5, 0)).quantile(1, dim="floor") * 1000,
        color="grey",
        alpha=0.2,
        linewidth=0,
    )
    for _, case_props in special_cases.items():
        try:
            plt.plot(
                medians[var].sel(height=slice(5, 0)).sel(floor=case_props["date"]).T
                * 1000,
                medians.height.sel(height=slice(5, 0)),
                color=case_props["color"],
                alpha=1,
                linewidth=1,
            )
        except KeyError:
            continue
    plt.plot(
        medians[var].sel(height=slice(5, 0)).median(dim="floor").T * 1000,
        medians.height.sel(height=slice(5, 0)),
        color="black",
        alpha=1,
        linewidth=1,
    )
    sns.despine()
    plt.savefig(f"../figures/profiles_3D_qc_{workflow_id}_mean.pdf", bbox_inches="tight")

    var = "cf"
    data = medians
    fig = plt.figure(figsize=(3, 4), dpi=200)
    plt.fill_betweenx(
        data.height.sel(height=slice(5, 0)),
        data[var].sel(height=slice(5, 0)).quantile(0.25, dim="floor"),
        data[var].sel(height=slice(5, 0)).quantile(0.75, dim="floor"),
        color="grey",
        alpha=0.2,
        linewidth=0,
    )
    plt.fill_betweenx(
        data.height.sel(height=slice(5, 0)),
        data[var].sel(height=slice(5, 0)).quantile(0, dim="floor"),
        data[var].sel(height=slice(5, 0)).quantile(1, dim="floor"),
        color="grey",
        alpha=0.2,
        linewidth=0,
    )
    for _, case_props in special_cases.items():
        try:
            plt.plot(
                data[var].sel(height=slice(5, 0)).sel(floor=case_props["date"]).T,
                data.height.sel(height=slice(5, 0)),
                color=case_props["color"],
                alpha=1,
                linewidth=1,
            )
        except KeyError:
            continue
    plt.plot(
        data[var].sel(height=slice(5, 0)).median(dim="floor").T,
        data.height.sel(height=slice(5, 0)),
        color="black",
        alpha=1,
        linewidth=1,
    )
    plt.xticks([0, 0.05, 0.1], labels=["0", "", "0.1"])
    maximum = np.round(
        data[var].sel(height=slice(5, 0)).median(dim="floor").max().item(0), 2
    )
    ax = plt.gca()
    ax.set_xticks([maximum], labels=[str(maximum)], minor=True)
    sns.despine()
    plt.savefig(f"../figures/profiles_3D_cf_{workflow_id}_mean.pdf", bbox_inches="tight")

    # +
    var = "wspd"
    fig = plt.figure(figsize=(3, 4), dpi=200)
    plt.fill_betweenx(
        data.height.sel(height=slice(5, 0)),
        data[var].sel(height=slice(5, 0)).quantile(0.25, dim="floor"),
        data[var].sel(height=slice(5, 0)).quantile(0.75, dim="floor"),
        color="grey",
        alpha=0.2,
        linewidth=0,
    )
    plt.fill_betweenx(
        data.height.sel(height=slice(5, 0)),
        data[var].sel(height=slice(5, 0)).quantile(0, dim="floor"),
        data[var].sel(height=slice(5, 0)).quantile(1, dim="floor"),
        color="grey",
        alpha=0.2,
        linewidth=0,
    )
    for _, case_props in special_cases.items():
        try:
            plt.plot(
                data[var].sel(height=slice(5, 0)).sel(floor=case_props["date"]).T,
                data[var].height.sel(height=slice(5, 0)),
                color=case_props["color"],
                alpha=1,
                linewidth=1,
            )
        except KeyError:
            continue
    plt.plot(
        data[var].sel(height=slice(5, 0)).median(dim="floor").T,
        data.height.sel(height=slice(5, 0)),
        color="black",
        alpha=1,
        linewidth=1,
    )
    plt.xticks(np.arange(0, 16, 2.5), labels=["0", "", "", "", "", "", "15"])
    maximum = np.round(
        data[var].sel(height=slice(5, 0)).median(dim="floor").max().item(0), 2
    )

    if include_obs:
        u_median = dropsondes.sel(alt=slice(10, 5000)).u.median(dim="launch_time")
        v_median = dropsondes.sel(alt=slice(10, 5000)).v.median(dim="launch_time")
        wspd_median = mcalc.wind_speed(
            u_median * munits.units["m/s"], v_median * munits.units["m/s"]
        ).pint.dequantify()
        plt.plot(
            wspd_median,
            dropsondes.sel(alt=slice(10, 5000)).alt / 1000,
            color="black",
            alpha=1,
            linewidth=1,
            linestyle="--",
        )

    ax = plt.gca()
    ax.set_xticks([maximum], labels=[str(maximum)], minor=True)
    sns.despine()
    plt.savefig(
        f"../figures/profiles_3D_wspd_{workflow_id}_mean.pdf", bbox_inches="tight"
    )
    # -
