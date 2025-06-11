from osgeo import gdal
import numpy as np





def imgread(path) -> np.ndarray:
    img = gdal.Open(path)
    c = img.RasterCount
    img_arr = img.ReadAsArray()
    if c > 1:
        img_arr = img_arr.swapaxes(1, 0)
        img_arr = img_arr.swapaxes(2, 1)
    del img
    return img_arr