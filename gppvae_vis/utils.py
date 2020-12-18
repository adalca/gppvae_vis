
# third party
import matplotlib
import numpy as np


def overlay_image_heatmaps(gray_im,
                           heatmap_im,
                           perc=[0.5, 0.5], 
                           cmap=None,
                           min_clip=None,
                           max_clip=None,
                           min_show=None,
                           max_show=None):

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
                                grid=None,
                                **kwargs):
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
