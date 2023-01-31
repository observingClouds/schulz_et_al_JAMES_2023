# Calculate TOA fluxes
import logging
import sys

import pandas as pd
import xarray as xr
from intake import open_catalog
from omegaconf import OmegaConf

sys.path.append("../helpers/")
import cluster_helpers as ch  # noqa: E402
import grid_helpers as gh  # noqa: E402

if __name__ == "__main__":
    client = ch.setup_cluster("local cluster", verbose=logging.ERROR)
    print(client)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler("../../logs/calc_TOAfluxes.log"),
            logging.StreamHandler(),
        ],
    )

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

    CRE_output_fmt = cfg.ANALYSIS.CRE.output_filename_fmt

    def calc_cre_sim(domain, cells, tqc_dia_threshold=0, tqi_dia_threshold=0.5):
        ds_rad = cat.simulations.ICON.LES_CampaignDomain_control[
            f"radiation_DOM0{domain}"
        ].to_dask()
        ds_sfc = cat.simulations.ICON.LES_CampaignDomain_control[
            f"surface_DOM0{domain}"
        ].to_dask()

        logging.info("Reindexing surface dataset")
        ds_sfc = ds_sfc.reindex(time=ds_rad.time, method="nearest")

        ds_sfc = ds_sfc.sel(time=slice("2020-01-11", "2020-02-18"), cell=cells)
        ds_rad = ds_rad.sel(time=slice("2020-01-11", "2020-02-18"), cell=cells)

        logging.info("Create calculation recipes")
        ds_rad["water_cloud_mask"] = ds_sfc.tqc_dia > tqc_dia_threshold
        ds_rad["ice_cloud_mask"] = ds_sfc.tqi_dia > tqi_dia_threshold

        mask_cloud_free = (ds_rad.water_cloud_mask == False) & (  # noqa: E712
            ds_rad.ice_cloud_mask == False  # noqa: E712
        )
        mask_cloudy = (ds_rad.water_cloud_mask == True) & (  # noqa: E712
            ds_rad.ice_cloud_mask == False  # noqa: E712
        )

        net_sw_cre = (
            ds_rad.sob_t.where(mask_cloud_free).mean(["cell"])
            - ds_rad.sob_t.where(mask_cloudy).mean(["cell"])
        ).resample(time="1D").mean() * -1

        net_lw_cre = (
            (
                ds_rad.thb_t.where(mask_cloud_free).mean(["cell"])
                - ds_rad.thb_t.where(mask_cloudy).mean(["cell"])
            )
            .resample(time="1D")
            .mean()
        )
        net_sw_cre.load()
        net_lw_cre.load()
        net_cre = net_sw_cre + net_lw_cre
        return net_sw_cre, net_lw_cre, net_cre

    ## Observations
    logging.info("Handling observations")
    ### Load data
    ceres = cfg.OBS.SATELLITES.CERES["SYN1deg-1H"].local
    ds_ceres = xr.open_dataset(ceres).sel(
        lat=slice(geobounds["lat_min"], geobounds["lat_max"]),
        lon=slice(
            (360 + geobounds["lon_min"]) % 360, (360 + geobounds["lon_max"]) % 360
        ),
        time=slice("2020-01-09", "2020-02-19"),
    )

    ceres_lw = ds_ceres.toa_lw_all_1h
    ceres_sw = ds_ceres.toa_sw_all_1h
    ceres_net = ds_ceres.toa_net_all_1h

    ### Calculate daily averages
    obs_net_daily = ceres_net.resample(time="1D").mean().mean(["lat", "lon"]).compute()

    ### Calculate CRE
    ceres_cre_lw = (
        (ds_ceres.toa_lw_clr_1h - ds_ceres.toa_lw_all_1h)
        .resample(time="1D")
        .mean()
        .mean(["lat", "lon"])
    )
    ceres_cre_sw = (
        (ds_ceres.toa_sw_clr_1h - ds_ceres.toa_sw_all_1h)
        .resample(time="1D")
        .mean()
        .mean(["lat", "lon"])
    )

    ## Simulation
    ### Load data
    logging.info("Handling simulation output")
    result_dict = {}
    for domain in [1, 2]:
        ds_rad = cat.simulations.ICON.LES_CampaignDomain_control[
            f"radiation_DOM0{domain}"
        ].to_dask()
        grid = cat.simulations.grids[ds_rad.uuidOfHGrid].to_dask()
        cells = gh.load_grid_subset(
            domain,
            [geobounds["lat_min"], geobounds["lat_max"]],
            [geobounds["lon_min"], geobounds["lon_max"]],
            grid,
            path="../../data/intermediate/",
        )

        ds_rad = ds_rad.sel(time=slice("2020-01-11", "2020-02-18"), cell=cells)
        net = ds_rad.sob_t + ds_rad.thb_t

        ### Calculate daily averages
        sim_net_daily = net.resample(time="1D").mean().mean(["cell"]).compute()
        result_dict[f"ICON_DOM{domain:02g}_net"] = sim_net_daily

        ### Calculate CRE
        (
            result_dict[f"ICON_DOM{domain:02g}_cre_sw"],
            result_dict[f"ICON_DOM{domain:02g}_cre_lw"],
            result_dict[f"ICON_DOM{domain:02g}_cre_net"],
        ) = calc_cre_sim(domain, cells)

    # Combine data
    logging.info("Preparing output")
    start_time = min(
        ceres_cre_sw.time.min(), result_dict["ICON_DOM02_cre_net"].time.min()
    )
    end_time = max(ceres_cre_sw.time.max(), result_dict["ICON_DOM02_cre_net"].time.max())
    reindex_dates = pd.date_range(start_time.values, end_time.values, freq="1D")
    output_dict = {
        "netCRE_daily_CERES": (ceres_cre_lw + ceres_cre_sw)
        .reindex(time=reindex_dates)
        .values,
        "lwCRE_daily_CERES": ceres_cre_lw.reindex(time=reindex_dates).values,
        "swCRE_daily_CERES": ceres_cre_sw.reindex(time=reindex_dates).values,
        "netCRE_daily_DOM01": result_dict["ICON_DOM01_cre_net"]
        .reindex(time=reindex_dates)
        .values,
        "lwCRE_daily_DOM01": result_dict["ICON_DOM01_cre_lw"]
        .reindex(time=reindex_dates)
        .values,
        "swCRE_daily_DOM01": result_dict["ICON_DOM01_cre_sw"]
        .reindex(time=reindex_dates)
        .values,
        "netCRE_daily_DOM02": result_dict["ICON_DOM02_cre_net"]
        .reindex(time=reindex_dates)
        .values,
        "lwCRE_daily_DOM02": result_dict["ICON_DOM02_cre_lw"]
        .reindex(time=reindex_dates)
        .values,
        "swCRE_daily_DOM02": result_dict["ICON_DOM02_cre_sw"]
        .reindex(time=reindex_dates)
        .values,
        "net_daily_CERES": obs_net_daily.reindex(time=reindex_dates).values,
        "net_daily_DOM01": result_dict["ICON_DOM01_net"]
        .reindex(time=reindex_dates)
        .values,
        "net_daily_DOM02": result_dict["ICON_DOM02_net"]
        .reindex(time=reindex_dates)
        .values,
    }

    # Create dataframe
    df = pd.DataFrame.from_dict(output_dict, orient="columns")
    df = df.set_index(reindex_dates)
    df.to_json(CRE_output_fmt)
