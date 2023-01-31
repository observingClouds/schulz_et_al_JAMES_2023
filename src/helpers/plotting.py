import cartopy.crs as ccrs
import cartopy.feature as cf
import datashader
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datashader.mpl_ext import dsshow
from matplotlib import ticker as mticker


def plot(da, grid, vmin=None, vmax=None, cmap="RdBu_r", dpi=100):
    # Lazy loading of output and grid

    central_longitude = 0  # -53.54884554550185
    # central_latitude = 12.28815437976341
    # satellite_height = 8225469.943160511

    projection = ccrs.PlateCarree(
        central_longitude=central_longitude
    )  # , central_latitude=central_latitude, satellite_height=satellite_height)

    coords = projection.transform_points(
        ccrs.Geodetic(),
        np.rad2deg(grid.clon.values),
        np.rad2deg(grid.clat.values),
    )

    fig, ax = plt.subplots(subplot_kw={"projection": projection}, dpi=dpi)
    fig.canvas.draw_idle()
    ax.add_feature(cf.COASTLINE, linewidth=0.8)

    gl = ax.gridlines(projection, draw_labels=True, alpha=0.35)
    gl.top_labels = False
    gl.right_labels = False
    gl.ylocator = mticker.FixedLocator(np.arange(11, 16, 1))
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 0, 1))

    ax.add_patch(
        mpatches.Circle(
            xy=[-57.717, 13.3],
            radius=1,
            edgecolor="grey",
            fill=False,
            transform=ccrs.PlateCarree(),
            zorder=30,
        )
    )

    artist = dsshow(
        pd.DataFrame(
            {
                "val": da.values,
                "x": coords[:, 0],
                "y": coords[:, 1],
            }
        ),
        datashader.Point("x", "y"),
        datashader.mean("val"),
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        ax=ax,
    )

    fig.colorbar(artist, label=f"{da.units}", shrink=0.8)
    return fig
