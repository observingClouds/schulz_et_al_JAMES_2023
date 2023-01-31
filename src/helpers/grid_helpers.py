"""grid helpers."""
import os
import pickle
import types

import numpy as np
from haversine import haversine_vector


def grid_selection(
    grid,
    lats=types.MappingProxyType({"lats": [11, 15]}),
    lons=types.MappingProxyType({"lons": [-59.3, -55.3]}),
):
    # Subsection
    x_range = lons
    y_range = lats

    # Create grid-mask
    cell = (
        (grid.clat.values >= np.deg2rad(y_range[0]))
        & (grid.clat.values <= np.deg2rad(y_range[1]))
        & (grid.clon.values >= np.deg2rad(x_range[0]))
        & (grid.clon.values <= np.deg2rad(x_range[1]))
    )
    grid = grid.sel(cell=cell)
    return cell, grid


def load_grid_subset(dom, lats, lons, grid, path=".", return_grid=False):
    pkl_filename = os.path.join(
        path,
        f"cells_DOM0{dom}_lats{'-'.join(map(str,lats))}_lons{'-'.join(map(str,lons))}.pkl",
    )
    if not os.path.exists(pkl_filename) or return_grid:
        print("Creating cell mask")
        cell_subsection, grid_subsection = grid_selection(grid, lats, lons)
        with open(pkl_filename, "wb") as f:
            pickle.dump(cell_subsection, f)
    else:
        print("Reading cell mask")
        with open(pkl_filename, "rb") as f:
            cell_subsection = pickle.load(f)
    if return_grid:
        return cell_subsection, grid_subsection
    else:
        return cell_subsection


def calc_haversine_distances(point, grid):
    """Calculate haversine distances between given point and all provided grid
    cells.

    Input
    -----
    point : tuple(lat, lon)
        point given in degree
    grid : xarray dataset
        horizontal (ICON) grid

    Returns
    -------
    distances : array
         distances between point and each cell
         given in degree
    """
    X = np.repeat(point, grid.dims["cell"]).reshape(2, grid.dims["cell"]).T
    Y = np.array([np.rad2deg(grid.clat.values), np.rad2deg(grid.clon.values)]).T
    distances = haversine_vector(X, Y, unit="deg")
    return distances


def get_cells_within_circle(grid, point, radius, preselection=True):
    """Select cells within a particular radius around a given point.

    Input
    -----
    grid : xarray dataset
        horizontal (ICON) grid
    point : tuple(lat, lon)
        point given in degree
    radius : float
        maximum distance between point
        and cells
    preselection : bool
        flag to automatically crop the grid
        to speed up computation of distances

    Returns
    -------
    cell_mask : boolean array
        True, where cell within radius, False otherwise
    """
    if preselection:
        min_lat, min_lon = np.array(point) - radius
        max_lat, max_lon = np.array(point) + radius

        cell_mask, grid = grid_selection(
            grid, lats=[min_lat, max_lat], lons=[min_lon, max_lon]
        )

    distances = calc_haversine_distances(point, grid)
    mask = np.zeros(len(grid.clat), dtype="bool")
    mask[distances <= radius] = True

    if preselection:
        cell_mask[cell_mask] = mask
        mask = cell_mask
    return mask
