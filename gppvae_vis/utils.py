
# third party
import matplotlib
import numpy as np
import pystrum as pm


def make_mosaic(imgs, rows, cols):
  i = 0
  mosaic = []
  for ir in range(rows):
    row = []
    for ic in range(cols):
      if i>=len(imgs):
        row.append(np.zeros_like(imgs[0]))
      else:
        row.append(imgs[i])
      i = i+1
    row = np.concatenate(row, 1)
    mosaic.append(row)
  mosaic = np.concatenate(mosaic, 0)
  return mosaic



def seg_overlap_rgb(vol, seg, do_contour=True, do_rgb=True, cmap=None, thickness=1.0):
    '''
    overlap a nd volume and nd segmentation (label map)
    do_contour should be None, boolean, or contour_type from seg2contour
    not well tested yet.
    '''

    # compute contours for each label if necessary
    if do_contour is not None and do_contour is not False:
        if not isinstance(do_contour, str):
            do_contour = 'inner'
        seg = pm.pynd.segutils.seg2contour(seg, contour_type=do_contour, thickness=thickness)

    # compute a rgb-contour map
    if do_rgb:
        if cmap is None:
            nb_labels = np.max(seg).astype(int) + 1
            colors = np.random.random((nb_labels, 3)) * 0.5 + 0.5
            colors[0, :] = [0, 0, 0]
        else:
            colors = cmap[:, 0:3]

        olap = colors[seg.flat, :]
        sf = seg.flat == 0
        for d in range(3):
          vd = vol[..., d]
          olap[sf, d] = vd.flat[sf]
        olap = np.reshape(olap, vol.shape)

    else:
        olap = seg
        olap[seg == 0] = vol[seg == 0]

    return olap

  

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
        gray_im (2D numpy array in [0, 1]): grayscale 2D image, [0, 1]
        heatmap_im (2d numpy array): heatmap image, can be any real number.
        perc (list, optional): weights of image. Need not add up to 1. Defaults to [0.2, 0.8]. 
        cmap (cmap function, optional): colormap function, such as those returned by 
            matplotlib.cm.get_cmap(). Defaults to None, which will lead to RdBu (red-blue) colormap
        min_clip (float, optional): smallest number to show on heatmap
            lower than min_clip will be clipped to min_clip. Defaults to None
        max_clip (float, optional): largest number to show on heatmap
            higher than max_clip will be clipped to max_clip. Defaults to None
        min_show (float, optional): the minimum values to show from center
            higher values than min_show will be 0'ed. Defaults to None.
        max_show (float, optional): the maximum values to show from center
            lower values than min_show will be 0'ed. Defaults to None.

    Returns:
        RGB numpy array: overlay of image and heatmap
    """

    # some input checking
    assert gray_im.ndim == 2, '2D images only, found: {}'.format(gray_im.ndim)
    assert heatmap_im.ndim == 2, '2D images only, found: {}'.format(heatmap_im.ndim)
    assert gray_im.min() >= 0, 'grayscale image should be in [0,1]'
    assert gray_im.max() <= 1, 'grayscale image should be in [0,1]'

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
        gray_images (list of 2D numpy arrays in [0, 1]): list of 2D grayscale images
        heatmap_images (list of 2d numpy arrrays): list of 2D heatmaps 
        grid (bool/2-element-list, optional): boolean (whether to have grid) or [row, col] 
            for grid dimensions. Defaults to True.

    Returns:
        RGB numpy array: mosaic of overlay of images and heatmaps (one *large* numpy array)
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
