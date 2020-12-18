
# third party
import matplotlib
import numpy as np


def overlay_image_heatmaps(gray_im,
                           heatmap_im,
                           perc=[0.2, 0.8], 
                           cmap=None,
                           min_clip=None,
                           max_clip=None,
                           min_show=None,
                           max_show=None):
    """
    Compute an image overlapping between the gray_im (2D) and heatmap_im (2D)

    Args:
        gray_im ([type]): grayscale 2D image, [0, 1]
        heatmap_im ([type]): heatmap image, can be any real number.
        perc (list, optional): weights of image. Need not add up to 1. Defaults to [0.2, 0.8]. 
        cmap ([type], optional): colormap function, such as those returned by matplotlib.cm.get_cmap() 
            Defaults to None, which will lead to RdBu (red-blue) colormap
        min_clip ([type], optional): smallest number to show on heatmap
            lower than min_clip will be clipped to min_clip. Defaults to None
        max_clip ([type], optional): largest number to show on heatmap
            higher than max_clip will be clipped to max_clip. Defaults to None
        min_show ([type], optional): the minimum values to show from center
            higher values than min_show will be 0'ed. Defaults to None.
        max_show ([type], optional): the maximum values to show from center
            lower values than min_show will be 0'ed. Defaults to None.

    Returns:
        [type]: [description]
    """

    # some input checking
    assert gray_im.ndim == 2, '2D images only, found: {}'.format(gray_im.ndim)
    assert heatmap_im.ndim == 2, '2D images only, found: {}'.format(heatmap_im.ndim)

    # prepare colormap
    if cmap is None:
        cmap = matplotlib.cm.get_cmap('RdBu')
    
    # process heatmap
    heatmap_im = heatmap_im.copy()
    if min_clip is not None: heatmap_im = np.clip(heatmap_im, min_clip, np.inf)
    if max_clip is not None: heatmap_im = np.clip(heatmap_im, -np.inf, max_clip)
    if min_show is not None: 
        assert max_show is not None, 'if min_show is not None'
        mask = np.logical_and(heatmap_im > min_show, heatmap_im < max_show)
        heatmap_im[mask] = 0
    
    # normalize for visualization
    heatmap_im = heatmap_im - heatmap_im.min()
    heatmap_im = heatmap_im / heatmap_im.max()
    
    # get colors
    col_im = cmap(heatmap_im.flatten())
    col_im = col_im[...,:3].reshape(heatmap_im.shape + (3, ))
    
    # prepare mixed volume
    gray_rgb = np.repeat(gray_im[..., np.newaxis], 3, -1)
    fin = gray_im[..., np.newaxis] * perc[0] + col_im * perc[1]

    # in any area that the heatmap is 0, keep the gray image rather than the mix
    if min_show is not None:
        mask_rgb = np.repeat(mask[..., np.newaxis], 3, -1)
        fin[mask_rgb] = gray_rgb[mask_rgb]
    
    # output
    return fin


def overlay_image_heatmaps_grid(gray_images,
                                heatmap_images,
                                grid=True,
                                **kwargs):
    """ 
    Compute a mosaid/grid of overlay images

    Args:
        gray_images ([type]): list of 2D grayscale images
        heatmap_images ([type]): list of 2D heatmaps 
        grid ([type], optional): boolean (whether to have grid) or [row, col] for grid dimensions.
            Defaults to True.

    Returns:
        [type]: [description]
    """

    nb_plots = len(gray_images)

    # figure out the number of rows and columns (code taken from ne.plot.slices)
    if grid:
        if isinstance(grid, bool):
            rows = np.floor(np.sqrt(nb_plots)).astype(int)
            cols = np.ceil(nb_plots/rows).astype(int)
        else:
            assert isinstance(grid, (list, tuple)), \
                "grid should either be bool or [rows,cols]"
            rows, cols = grid
    else:
        rows = 1
        cols = nb_plots
    
    # prepare one large volume
    shp = gray_images[0].shape
    mosaic_shape = np.array(shp + (1,)) * np.array((rows, cols, 3))
    mosaic = np.zeros(mosaic_shape)

    # go through all slices
    for p in range(nb_plots):
        col = np.remainder(p, cols)
        row = np.floor(p/cols).astype(int)
        olay = overlay_image_heatmaps(gray_images[p], heatmap_images[p], **kwargs)
        mosaic[row*shp[0]:(row+1)*shp[0], col*shp[1]:(col+1)*shp[1], :] = olay
    
    return mosaic
