"""Cloud Cluster Analysis Module.

This module contains functions for cloud cluster identification and
analysis
"""

import numpy as np
from haversine import haversine
from scipy import stats
from scipy.spatial import distance


def add_boundary_to_cloud_slice(c_slice):
    """Expand cloud slice by two rows/colums of zeros.

    Enables to calculate the gradient and therefor detect
    the edges of the field (1 extra row) and also in case
    the edge is directly on the original (c_slice) field.

    Note: in principle the extra columns are not necessary to
    add but that is the standard behaviour of np.pad

    >>> add_boundary_to_cloud_slice(np.array([[2,2,2],[2,2,2]]))
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 2, 2, 2, 0, 0],
           [0, 0, 2, 2, 2, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]])
    """
    import numpy as np

    return np.pad(c_slice, 2, mode="constant")


def f_cloud_edges(c_slice, verbose=False):
    """Algorithm to detect cloud base and cloud top.

    For a layered cloud like
    array([[0., 3., 3., 3.],
       [0., 3., 3., 3.],
       [0., 0., 0., 0.],
       [3., 3., 3., 3.],
       [0., 3., 0., 3.]])

    the highest and lowest idx are returned.

    In case of cloud column gaps the returned index is "nan".

    >>> f_cloud_edges(np.array([[0., 3., 0., 3.],\
       [0., 3., 0., 3.],\
       [3., 3., 0., 3.],\
       [3., 3., 0., 3.],\
       [0., 3., 0., 3.]]))
    array([[ 3.,  4., nan,  4.],
           [ 2.,  0., nan,  0.]])
    """

    import numpy as np

    height, width = np.shape(c_slice)

    # Add boundary of zeros to calculate gradient on the edges
    bounded_cloud = add_boundary_to_cloud_slice(c_slice)
    # Calculate gradient/difference between single items along hgt-axis
    diff = np.diff(bounded_cloud, axis=0)
    # First maximum difference in column is the cloud top
    top_idx = np.nanargmax(diff, axis=0)[2:-2]  # exclude padding width(left and right)
    # Last minimum difference is the cloud base
    #  reverse index listing to find the last index easier
    base_idx_reverse = np.nanargmin(diff[::-1, :], axis=0)[
        2:-2
    ]  # counting from below to get lowest cbh in case of layer

    # Exclude indices which are 0 or height in the padded version
    #  these are non-cloudy columns
    base_idx_reverse = np.where(base_idx_reverse == 0, np.nan, base_idx_reverse)
    top_idx = np.where(top_idx == 0, np.nan, top_idx)

    base_idx = height - base_idx_reverse  # counting from the top

    # top_idx = np.nanmax([top_idx,np.zeros((width,))],axis=0) #remove upper padding (but keep 0)
    # base_idx = np.nanmax([base_idx-1,np.zeros((width,))],axis=0)

    top_idx = top_idx - 1

    if verbose:
        print("Cloud slice")
        print(c_slice)
        print("Added boundaries for easier calculation")
        print(bounded_cloud)
        print("Difference between column items for border detection")
        print(diff)
        print("Indices")
        print(base_idx, top_idx)

    return np.array([base_idx, top_idx])


def detect_precip(c_slice, lowest_idx, threshold=3):
    """

    Input
    lowest_idx: lowest height index of the original grid
        necessary to put the c_slice values into their
        context.
        As precip is present in the case the lowest two
        range gates show a return signal, the lowest two
        range gates of the c_slice by themselves could
        also be in the high troposphere. Therefore the lowest_idx
    """

    import numpy as np

    if lowest_idx < threshold:
        precip_idx = np.where(np.all(~np.isnan(c_slice[0:2, :]), axis=0), 1, 0)
        return precip_idx


def f_apply_cloudtype(label, dictionary):
    import numpy as np

    if np.isnan(label):
        return np.nan
    else:
        return dictionary[label]


def fractal_dimension(Z, threshold=290):
    """Calculates Minkowski-Bouligand dimension of a given data array.

    Parameters
    ----------
    Z : array_like
        Input data array
    theshold : scalar, optional
        Threshold for converting Z into a binary array

    Returns
    -------
    dimension : scalar
        Minkowski-Bouligand dimension

    Origin
    ------
    https://stackoverflow.com/questions/44793221/python-fractal-box-count-fractal-dimension
    """
    import numpy as np
    import scipy

    # Only for 2d image
    assert len(Z.shape) == 2

    # From https://github.com/rougier/numpy-100 (#87)
    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
            np.arange(0, Z.shape[1], k),
            axis=1,
        )

        # We count non-empty (0) and non-full boxes (k*k)
        return len(np.where((S > 0) & (S < k * k))[0])

    # Transform Z into a binary array
    Z = Z < threshold

    # Minimal dimension of image
    p = min(Z.shape)

    # Greatest power of 2 less than or equal to p
    n = 2 ** np.floor(np.log(p) / np.log(2))

    # Extract the exponent
    n = int(np.log(n) / np.log(2))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2 ** np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))

    # Fit the successive log(sizes) with log (counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]


def prepare_labels(labels):
    labels_0 = np.where(np.logical_or((np.isnan(labels)), (labels < 0)), 0, labels)
    unique_labels = np.unique(labels)[1:]
    return unique_labels, labels_0


def shifting(arr, gradient):
    import scipy.cluster.vq as scv

    code, dist = scv.vq(arr, gradient)
    return code, dist


def imagecolor2data(img_arr, data_range, cmap):
    """Converts RGB color values to data values.

    Image data contains only RGB values,
    but if the colormap and the original
    data range is know, the data information
    can be retrieved again.

    Source
    ------
    adapted from http://stackoverflow.com/questions/3720840/
    how-to-reverse-color-map-image-to-scalar-values/3722674#3722674
    """
    import matplotlib.cm as cm
    import matplotlib.colors
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.cluster.vq as scv

    if isinstance(cmap, str):
        cmap = eval("cm." + cmap)
        gradient = cmap(np.linspace(0.0, 1.0, len(data_range)))
    elif isinstance(cmap, np.ndarray):
        gradient = cmap
    elif isinstance(cmap, matplotlib.colors.ListedColormap):
        gradient = cmap(np.linspace(0.0, 1.0, len(data_range)))

    # Reshape arr to something like (240*240, 4), all the 4-tuples in a long list...
    arr2 = img_arr.reshape((img_arr.shape[0] * img_arr.shape[1], img_arr.shape[2]))

    # Use vector quantization to shift the values in arr2 to the nearest point in
    # the code book (gradient).
    #     code,dist=scv.vq(arr2,gradient)
    code, dist = shifting(arr2, gradient)

    # code is an array of length arr2 (240*240), holding the code book index for
    # each observation. (arr2 are the "observations".)
    # Scale the values so they are from 0 to 1.
    values = code.astype("float") / gradient.shape[0]

    # Reshape values back to (240,240)
    values = values.reshape(img_arr.shape[0], img_arr.shape[1])
    values = values[::-1]

    # Transform the values from 0..1 to actual data
    lookup_table = dict(
        zip(np.linspace(0, 1, len(data_range) + 1).round(3), data_range, strict=True)
    )
    data_values = np.vectorize(lookup_table.get)(values.round(3))

    return data_values


## Iorg and others
def nearest_neighbour(points, output_nn=False):
    """Calculates for a given list of points (x,y) in a 2D space the nearest
    point.

    Returns the nearest neighbour distances and optionally two lists
    containing the original point and its neighbour.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    NN_distance = np.empty((len(points)))
    if output_nn:
        NN_points = np.empty((len(points), 2))

    for i, point in enumerate(points):
        if any(np.isnan(point)):
            continue
        distance, ind = tree.query(point, k=2)
        NN_distance[i] = distance[1]
        try:
            closestPoint = points[ind[1]]
        except IndexError:
            continue
        if output_nn:
            NN_points[i] = closestPoint

    if output_nn:
        return NN_distance, NN_points
    else:
        return NN_distance


def NNCDF_ran(npa, r=None, rmax=1, resolution=1000):
    """
    npa: lambda = number of points / area
    """
    import numpy as np

    if r is None:
        r = np.linspace(0, rmax, resolution)
    return 1 - np.exp(-npa * np.pi * r**2)


def label_cluster(cluster_mask, stencil=None, undilute=True):
    """Find coherent clusters.

    Method to find and label coherent clusters.
    The stencil can be used to count also masked points
    which are not direct neighbours to the cluster.

    Input
    -----
    cluster_mask : array-like
        Two dimensional binary array being True where a cluster
        exists and False otherwise

    stencil : np.ones(N,N)
        Kernel/stencil to check connectivities.
        N should be uneven, because otherwise the stencil
        is not centered

    undilute : boolean
        If True (default), then only the original cluster_mask
        positions are labeled, otherwise also the labels for the
        diluted fieled are included

    Returns
    -------
    cluster_labels : np.array, same shape as cluster_mask
        Array where connected clusters are given one number starting at 1.
        Background is 0.

    Example
    -------
    >>> a = np.array([[1., 0.,0, 0., 1.],
       [1., 0., 0, 0., 0.],
       [1., 0., 0, 1., 1.],
       [1., 0, 0, 1., 1.]])
    >>> label_cluster(a,stencil=np.ones((1,1)))
    array([[1, 0, 0, 0, 2],
       [1, 0, 0, 0, 0],
       [1, 0, 0, 3, 3],
       [1, 0, 0, 3, 3]], dtype=int32)
    >>> label_cluster(a,stencil=np.ones((3,3)), undilute=False)
    array([[1, 1, 0, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1]], dtype=int32)
    >>> label_cluster(a,stencil=np.ones((3,3)))
        np.array([[1, 0, 0, 0, 0, 2],
       [1, 0, 0, 0, 0, 0],
       [1, 0, 0, 0, 2, 2],
       [1, 0, 0, 0, 2, 2]], dtype=int32)
    >>> label_cluster(a, stencil=np.array([[0,1,0],[1,1,1],[0,1,0]]),undilute=False)
    array([[1, 1, 0, 0, 2, 2],
       [1, 1, 0, 0, 2, 2],
       [1, 1, 0, 2, 2, 2],
       [1, 1, 0, 2, 2, 2]], dtype=int32)
    """
    import numpy as np
    from scipy.ndimage import binary_dilation, label

    if stencil is None:
        stencil = np.ones((1, 1))

    labels = label(binary_dilation(cluster_mask, structure=stencil))[0]
    # Undilute
    if undilute:
        labels[cluster_mask == 0] = 0

    return labels


def centroid(cluster_mask, cluster_labels):
    """Find centroid of cluster.

    Input
    -----
    cluster_mask : array-like
        Two dimensional binary array being True where a cluster
        exists and False otherwise

    cluster_labels : np.array, same shape as cluster_mask
        Array where connected clusters are given one number starting at 1.
        Background is 0.

    Returns
    -------
    centroids : np.array
        centroids as tuples of floating indices
        (x-axis, y-axis)

    Example
    -------
    >>> a = np.array([[1., 0, 0.,0, 0., 1.],
       [1., 0.,0, 0, 0., 0.],
       [1., 0.,0, 0, 1., 1.],
       [1., 0, 0,0, 1., 1.]])
    >>> l = np.array([[1, 0, 0, 0, 0, 2],
       [1, 0, 0, 0, 0, 0],
       [1, 0, 0, 0, 2, 2],
       [1, 0, 0, 0, 2, 2]], dtype=int32)
    >>> centroid(a,l)
    array([[1.5, 0. ],
       [2. , 4.6]])
    """
    import numpy as np
    from scipy.ndimage.measurements import center_of_mass as center

    labels2query = np.unique(
        cluster_labels[cluster_labels != 0]
    )  # label=0 is the background
    return np.asarray(center(cluster_mask, cluster_labels, labels2query))


def calc_Iorg(distances, domain_size):
    """Calculating the Iorg index."""
    import numpy as np

    n = len(distances)
    return 1 - np.sum(NNCDF_ran(n / domain_size**2, r=np.sort(distances))) / n


def calc_Iorg_complete(
    data,
    thresholds,
    dshape=None,
    mode="normal",
    stencil=None,
    return_centroids=False,
    return_size=False,
):
    """Calculating the Iorg index.

    Parameters
    ----------
    data : array-like
        input data array, e.g. brightness temperature,
        outgoing longwave radiation, column moisture
        if data is flattened, the dshape parameter has to be defined
    thresholds : tuple, (minimum, maximum)
        thresholds to mask the data to extract cloud clusters.
        Data between minimum and maximum is regarded as clouds.
        In case only one threshold is needed, the other should be set to
        -inf or +inf or any number which is smaller/larger as the data
        minimum/maximum
    dshape : tuple, optional
        in case data is flattened (this might be the case
        by applying the function with ndimage.generic_filter)
    mode : string, {"normal","GCM"}, optional
        define mode of Iorg calculation. In case of GCMs
        the option "GCM" might be chosen to handle each gridcell as
        an indiviual cluster (several gridcells are less likely to
        be one close cluster); default: "normal"

    Returns
    -------
    Iorg : scalar
        Organization index

    Note
    ----
    It might be necessary to adapt this function in case parts of the
    domain should be excluded e.g. when deep convection is present,
    although one is interested in shallow convection only. The threshold
    may mask it, but left an unnatrual space with no clouds at all.
    """
    import numpy as np

    if stencil is None:
        stencil = np.ones((3, 3))

    if dshape is not None:
        data = data.reshape(dshape)
    domain_size = np.sqrt(data.shape[0] * data.shape[1]) // 1
    cluster_mask = np.where((data > thresholds[0]) & (data < thresholds[1]), 1, 0)
    if ~np.any(cluster_mask):
        results = [np.nan]
        if return_centroids:
            results.append([np.nan])
        if return_size:
            results.append([np.nan])
        return results
    if mode == "normal":
        cluster_labels = label_cluster(cluster_mask, stencil)
    elif mode == "GCM":
        # Give every non-zero mask entry a new number/label
        cluster_labels = np.zeros_like(cluster_mask, dtype="int")
        cluster_labels[np.nonzero(cluster_mask)] = np.arange(
            1, len(np.nonzero(cluster_mask)[0]) + 1
        )
    points = centroid(cluster_mask, cluster_labels).round(0).astype(int)
    distances = nearest_neighbour(points)

    results = [calc_Iorg(distances, domain_size)]

    if return_size:
        mean_size = calc_cluster_size(cluster_mask, cluster_labels, normalize=True)
        results.append(mean_size)

    if return_centroids:
        results.append(points)

    return results


def calc_cluster_size(cluster_mask, cluster_labels, normalize=True):
    """Calculate the cluster size.

    Input
    -----
    cluster_mask : array-like
        Two dimensional binary array being True where a cluster
        exists and False otherwise

    cluster_labels : np.array, same shape as cluster_mask
        Array where connected clusters are given one number starting at 1.
        Background is 0.

    normalize : boolean
        If normalize is True (default) than the cluster size is
        given as a fraction of the domain, otherwise it is
        the count of clustery pixels

    Returns
    -------
    cluster_size : float
        Area or fraction of each single cluster
    """
    import numpy as np
    from scipy.ndimage.measurements import sum as cluster_sum

    sum = cluster_sum(cluster_mask, cluster_labels, np.unique(cluster_labels)[:-1])
    if normalize:
        sum = sum / (cluster_mask.shape[0] * cluster_mask.shape[1])
    return sum


def ecdf(cloud_lon, cloud_lat, metric, return_distances=False):
    """Calculate ecdf for given cloud centroids.

    Inputs
    ------
    cloud_lon, cloud_lat : 1D array like
        List of cloud positions either as indices
        or as longitudes and latitudes
    metric : 'euclidean' or 'haversine'
        For latitutudes and longitudes choose
        'haversine' to include the effect of
        curvature. In case of an equidistant grid,
        the 'euclidean' metric is sufficient

    Returns
    -------
    x_values, y_values
    """

    cloudcentres = np.vstack((cloud_lon, cloud_lat)).T
    if metric == "haversine":
        Y = distance.pdist(cloudcentres, haversine)
    elif metric == "euclidean":
        Y = distance.pdist(cloudcentres, "euclidean")
    Z = distance.squareform(Y)
    Z = np.ma.masked_where(Z == 0, Z)
    mindistances = np.min(Z, axis=0)

    max(mindistances)
    min(mindistances)

    # create a sorted series of unique data
    cdfx = np.sort(mindistances)
    # x-data for the ECDF: evenly spaced sequence of the uniques
    x_values = np.linspace(start=min(cdfx), stop=max(cdfx), num=len(cdfx))

    # size of the x_values
    mindistances.size
    # y-data for the ECDF:
    y_values = np.zeros(len(x_values))
    for i, x_value in enumerate(x_values):
        # all the values in raw data less than the ith value in x_values
        temp = mindistances[mindistances <= x_value]
        # fraction of that value with respect to the size of the x_values
        value = np.float64(len(temp)) / np.float64(len(mindistances))
        # pushing the value in the y_values
        y_values[i] = value
    # return both x and y values
    if return_distances:
        return x_values, y_values, mindistances
    else:
        return x_values, y_values


def Iorg_poisson(
    cloud_lon, cloud_lat, domainsize, metric="euclidean", return_distances=False
):
    """Calculate Iorg.

    Inputs
    ------
    cloud_lon, cloud_lat : 1D array like
        List of cloud positions either as indices
        or as longitudes and latitudes

    domainsize : integer
        Size of domain (len(x)*len(y))
        Note: in case of the 'haversine' metric
        the domainsize might be differently
        defined

    metric : 'euclidean' or 'haversine'
        For latitutudes and longitudes choose
        'haversine' to include the effect of
        curvature. In case of an equidistant grid,
        the 'euclidean' metric is sufficient

    Returns
    -------
    Iorg : float
    """
    nclouds_real = np.float(len(cloud_lon))

    if return_distances:
        x_values_ecdf, y_values_ecdf, mindistances = ecdf(
            cloud_lon, cloud_lat, metric, return_distances=True
        )
    else:
        x_values_ecdf, y_values_ecdf = ecdf(
            cloud_lon, cloud_lat, metric, return_distances=False
        )

    poisson = np.zeros(len(cloud_lon))
    r = x_values_ecdf
    density = nclouds_real / domainsize

    for e, element in enumerate(r):
        poisson[e] = 1 - np.exp(-density * np.pi * element**2)

    Iorg = np.trapz(y_values_ecdf, poisson)
    if return_distances:
        return Iorg, mindistances
    else:
        return Iorg


def calc_D0(cloudcenters, metric="euclidean"):
    # distance between points (center of mass of clouds) in pairs
    di = distance.pdist(cloudcenters, metric)
    # order-zero diameter
    D0 = stats.mstats.gmean(di)
    return D0


def scai(cloud_lon, cloud_lat, cluster_mask, connectivity=1, metric="euclidean"):
    """Calculates the 'Simple Convective Aggregation Index (SCAI)'.

    The SCAI is defined as the ratio of convective disaggregation
    to a potential maximal disaggregation.

    Input
    -----
    centroids : (list[:class:`RegionProperties`]):
        Output of function :func:`get_cloudproperties`

    cloudmask : ndarray
        2d binary cloud mask

    connectivity : int
        Maximum number of orthogonal hops to consider
        a pixel/voxel as a neighbor (see :func:`skimage.measure.label`)

    mask :array
        2d mask of non valid pixels.

    Returns
    -------
    SCAI : float
    """

    centroids = np.vstack((cloud_lon, cloud_lat)).T

    # number of cloud clusters
    N = len(centroids)

    # potential maximum of N depending on cloud connectivity
    if connectivity == 1:
        chessboard = np.ones(cluster_mask.shape)
        # assign every second element with "0"
        chessboard.flat[slice(1, None, 2)] = 0
        # inlcude NaN mask
        chessboard[np.isnan(cluster_mask)] = np.nan
        N_max = np.nansum(chessboard)
    elif connectivity == 2:
        chessboard[np.arange(1, cluster_mask.shape[0], 2), :] = 0
        chessboard = np.reshape(chessboard, cluster_mask.shape)
        chessboard[np.isnan(cluster_mask)] = np.nan
        N_max = np.nansum(chessboard)
    else:
        raise ValueError("Connectivity argument should be `1` or `2`.")

    D0 = calc_D0(centroids, metric)
    # characteristic length of the domain (in pixels): diagonal of box
    L = np.sqrt(cluster_mask.shape[0] ** 2 + cluster_mask.shape[1] ** 2)

    return (N / N_max) * (D0 / L) * 1000, D0


def cluster_analysis(
    data, thresholds, stencil=None, metric="euclidean", lats=None, lons=None
):
    """Calculates cluster properties.

    Properties are total number of clusters, area of clusters, distance
    between clusters (D0), Iorg and SCAI.
    """

    if stencil is None:
        stencil = np.ones((3, 3))

    domain_size = data.shape[0] * data.shape[1]
    cluster_mask = np.where((data > thresholds[0]) & (data < thresholds[1]), 1, 0)
    cluster_labels = label_cluster(cluster_mask, stencil=stencil)
    centroids = centroid(cluster_mask, cluster_labels)
    points = centroids.round(0).astype(int)
    cloud_lat_idx, cloud_lon_idx = points.T
    nclouds = len(cloud_lon_idx)

    # Iorg
    if metric == "euclidean":
        Iorg, distances = Iorg_poisson(
            cloud_lon_idx, cloud_lat_idx, domain_size, metric, return_distances=True
        )
    elif metric == "haversine":
        cloud_lon = lons[cloud_lon_idx]
        cloud_lat = lats[cloud_lat_idx]
        Iorg, distances = Iorg_poisson(
            cloud_lon, cloud_lat, domain_size, metric, return_distances=True
        )

    # Cluster size
    cluster_size = calc_cluster_size(cluster_mask, cluster_labels)

    # SCAI
    SCAI, D0 = scai(cloud_lon_idx, cloud_lat_idx, cluster_mask)

    return nclouds, Iorg, D0, np.nanmean(cluster_size), SCAI


def visualize_NN(origin=None, neighbours=None, mask=None, **kwargs):
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Initialize figure
    sns.set(style="ticks")

    if "ax" in kwargs:
        ax = kwargs.get("ax")
    else:
        _, ax = plt.subplots(1, 1, figsize=(10, 10))

    if mask is not None:
        ax.contourf(mask, cmap="Greys")

    # Plot the original points
    if origin is not None:
        ax.scatter(origin[:, 1], origin[:, 0], color="red")

    # Plot lines to nearest neighbours
    if neighbours is not None:
        for i in range(len(origin)):
            ax.plot([origin[i, 1], neighbours[i, 1]], [origin[i, 0], neighbours[i, 0]])

    sns.despine(left=True, bottom=True)
    plt.tick_params(
        axis="both",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )  # labels along the bottom edge are off
    #     plt.show()
    return


def visualize_distibution(distances, domain_size, cdf_vs_cdf=True):
    """"""
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Initialize figure
    sns.set(style="ticks")
    plt.figure(figsize=(5, 5))

    # Calculate theoretical distribution
    n = len(distances)
    rmax = np.nanmax(distances)
    distr_theoretical = NNCDF_ran(n / domain_size**2, rmax=np.ceil(rmax))

    if cdf_vs_cdf is False:
        # Plot normal distribution plots
        plt.plot(
            np.linspace(0, np.ceil(rmax), 1000),
            distr_theoretical,
            label="weibul computed",
        )
        plt.plot(np.sort(distances), np.array(range(n)) / float(n), label="real distr.")
        plt.xlabel("distance between points (index)")
        plt.ylabel("CDF")
        plt.legend()
    elif cdf_vs_cdf is True:
        plt.plot(
            NNCDF_ran(n / domain_size**2, r=np.sort(distances)), np.linspace(0, 1, n)
        )
        plt.plot([0, 1], [0, 1])
        plt.annotate("regular", (0.7, 0.1), fontsize=15)
        plt.annotate("organized", (0.1, 0.9), fontsize=15)

    sns.despine()
    return None


vf_apply_cloudtype = np.vectorize(f_apply_cloudtype, otypes=[np.float])


def get_organization_info(data, settings, verbose=False):
    """

    Input
    -----
    settings : dict
        example: settings = {'threshold_discard_percentile':25,
            'threshold_discard_temperature':285,
            'threshold_cluster_llimit':280,
            'threshold_cluster_ulimit':290,
            'stencil':np.ones((3,3))
           }
    """
    data_area = data.shape[0] * data.shape[1]

    # Check for deep disturbing convection
    # by excluding scenes, when the 25th percentile of brightness temperatures
    # is above/equal a threshold (285K)
    # EXCLUSION IS NOT DONE AT THIS STEP!
    percentile_data = np.nanpercentile(data, settings["threshold_discard_percentile"])

    # Create mask for finding cluster
    # Set data above threshold to 1 and otherwise to zero
    cluster_mask = np.where(
        (data > settings["threshold_cluster_llimit"])
        & (data < settings["threshold_cluster_ulimit"]),
        1,
        0,
    )

    # Label cluster
    labels = label_cluster(cluster_mask, settings["stencil"])
    if len(np.unique(labels)) <= 1:
        return [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

    # Find centroid
    cluster_center = centroid(data, labels).round(0).astype(int)

    # Find nearest neigbour distance
    cloud_NN_distances = nearest_neighbour(cluster_center)

    # Calculate Iorg value
    Iorg = calc_Iorg(cloud_NN_distances, np.sqrt(data_area)).round(2)

    # Calculate cluster size
    cluster_size = calc_cluster_size(cluster_mask, labels)

    # Calculate cloud fraction
    # cloud_fraction = np.count_nonzero(cluster_mask)/len(cluster_mask.flatten())
    cloud_fraction = np.count_nonzero(cluster_mask) / (
        len(cluster_mask.flatten()) - np.sum(data < 280)
    )
    # Calculate number of clusters
    N = len(cluster_center)

    return [
        Iorg,
        np.nanmean(cluster_size).round(6),
        np.nanstd(cluster_size).round(4),
        N,
        cloud_fraction,
        percentile_data,
    ]
