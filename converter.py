#code to turn swc files into png! ideally
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.draw import disk

def swc_to_tiff(test_file, file_dir, save_dir = "tiff_images/", pixels = 512):
    data = pd.read_csv(
    file_dir + test_file,
    comment='#',            # drop all lines starting with “#”
    sep='\s+',  # split on any run of spaces/tabs
    header=None,            # there’s no header row in the data
    names=[                 # assign the columns you want
        'ID',
        'BranchID',
        'x', 'y', 'z',
        'radius',
        'ParentID'
    ]
    )
    img = points_to_image(data, image_size = (pixels, pixels))
    plt.imsave(save_dir + test_file + ".tiff", img, vmin = 0, vmax = 255, cmap='gray', format = 'tiff')


def points_to_image(df, image_size):
    # extract arrays
    xs = df['x'].values
    ys = df['y'].values
    rs = df['radius'].values

    # compute data bounds
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()

    H, W = image_size

    # uniform scale to fit all points in [0, W-1]/[0, H-1]
    sx = (W - 1) / (xmax - xmin) if xmax > xmin else 1.0
    sy = (H - 1) / (ymax - ymin) if ymax > ymin else 1.0
    scale = min(sx, sy)
    # map real‐world coords → pixel coords
    x_pix = ((xs - xmin) * scale).astype(int)
    y_pix = ((ys - ymin) * scale).astype(int)
    r_pix = (rs * scale)

    # create blank image
    img = np.zeros((H, W), dtype=np.uint8)

    # draw each circle
    for cx, cy, rad in zip(x_pix, y_pix, r_pix):
        if rad > 0:
            rr, cc = disk((cy, cx), rad, shape=img.shape)
            img[rr, cc] = 255  # white disk on black bg

    return img


